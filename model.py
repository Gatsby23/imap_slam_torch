import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import time

#from visualizer import Visualizer
#import mcubes


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))


class IMAP(nn.Module):
    def __init__(self):
        super(IMAP, self).__init__()
        # Learn a positional embedding from x,y,z to 93 features
        self.positional_embedding = nn.Linear(3, 93, bias=False).cuda()
        nn.init.normal_(self.positional_embedding.weight, 0.0, 25.0)

        # NeRF model with 4 hidden layers of size 256
        self.fc1 = nn.Linear(93,256).cuda()
        self.fc2 = nn.Linear(256,256).cuda()
        self.fc3 = nn.Linear(256+93,256).cuda()
        self.fc4 = nn.Linear(256,256).cuda()
        self.fc5 = nn.Linear(256,4, bias=False).cuda()
        self.fc5.weight.data[3,:]*=0.1


    def forward(self, pos):
        # Position embedding uses a sine activation function
        position_embedding = torch.sin(self.positional_embedding(pos))

        # NeRF model
        x = F.relu(self.fc1(position_embedding))
        x = torch.cat([F.relu(self.fc2(x)), position_embedding], dim=1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        # Output is a 4D vector (r,g,b, density)
        out = self.fc5(x)

        return out


class Camera():
    def __init__(self, rgb_image, depth_image, 
                 position_x, position_y, position_z, 
                 rotation_x, rotation_y, rotation_z, 
                 light_scale=0.0, light_offset=0.0, 
                 focal_length_x=525.0, focal_length_y=525.0, 
                 principal_point_x=319.5, principal_point_y=239.5):

        # Camera parameters
        self.params = torch.tensor([rotation_x, rotation_y, rotation_z, 
                                    position_x, position_y, position_z, 
                                    light_scale, light_offset]).detach().cuda().requires_grad_(True)
        
        # Camera calibration parameters
        self.focal_length_x = focal_length_x
        self.focal_length_y = focal_length_y
        self.principal_point_x = principal_point_x
        self.principal_point_y = principal_point_y

        # Camera intrinsic matrix and its inverse
        self.K = torch.tensor([
            [focal_length_x, 0.0, principal_point_x],
            [0.0, focal_length_y, principal_point_y],
            [0.0, 0.0, 1.0],
            ], device='cuda', dtype=torch.float32, requires_grad=False)

        self.K_inverse = torch.tensor([
            [1.0/focal_length_x, 0.0, -principal_point_x/focal_length_x],
            [0.0, 1.0/focal_length_y, -principal_point_y/focal_length_y],
            [0.0, 0.0, 1.0],
            ], device='cuda', dtype=torch.float32, requires_grad=False)

        # Conversion factor for depth from 16bit color
        self.depth_conversion_factor = 1 / 50000.0

        # RGB and depth images
        self.set_images(rgb_image, depth_image)

        self.exp_a = torch.cuda.FloatTensor(1)
        self.rotation_matrix = torch.zeros(3,3, device='cuda')
        self.translation_matrix = torch.zeros(3,3, device='cuda')
        self.grid_sampling_probs = torch.full((64,), 1.0/64, device='cuda')
        self.image_size = depth_image.shape

        # Update transformation matrix
        self.update_transform()

        # Optimizer for camera parameters
        self.optimizer = optim.Adam([self.params], lr=0.005)

    def set_images(self, rgb_image, depth_image):
        self.rgb_image = torch.from_numpy((rgb_image).astype(np.float32)).cuda() / 256.0
        self.depth_image = torch.from_numpy(depth_image.astype(np.float32)).cuda() * self.depth_conversion_factor

    def update_transform(self):
        # Update transformation matrix based on camera parameters

        identity_matrix = torch.eye(3, device='cuda')

        # Create skew symmetric matrices
        skew_symmetric_matrices = [torch.zeros((3,3), device='cuda') for _ in range(3)]
        skew_symmetric_matrices[0][1, 2] = -1
        skew_symmetric_matrices[0][2, 1] = 1
        skew_symmetric_matrices[1][2, 0] = -1
        skew_symmetric_matrices[1][0, 2] = 1
        skew_symmetric_matrices[2][0, 1] = -1
        skew_symmetric_matrices[2][1, 0] = 1

        # Compute the norm of the rotation vector (gets the angle of rotation)
        rotation_norm = torch.norm(self.params[0:3])

        # Compute the inverse of the rotation norm 
        rotation_norm_inverse = 1.0 / (rotation_norm + 1e-12)

        # Normalize the rotation vector
        rotation_vector = rotation_norm_inverse * self.params[0:3]

        # Compute sine and cosine of the rotation for the rotation matrix
        cos_theta = torch.cos(rotation_norm)
        sin_theta = torch.sin(rotation_norm)

        # Compute the skew symmetric matrix for the rotation vector
        skew_symmetric_matrix = rotation_vector[0]*skew_symmetric_matrices[0] + rotation_vector[1]*skew_symmetric_matrices[1] + rotation_vector[2]*skew_symmetric_matrices[2]

        # Compute the square of the skew symmetric matrix
        skew_symmetric_matrix_squared = torch.matmul(skew_symmetric_matrix, skew_symmetric_matrix)

        # Compute the rotation matrix using the Rodrigues' rotation formula
        rotation_matrix = identity_matrix + sin_theta * skew_symmetric_matrix + (1.0 - cos_theta) * skew_symmetric_matrix_squared

        self.rotation_matrix = rotation_matrix
        self.translation_matrix = self.params[3:6]

        # Compute the exponential of the lighting parameter a
        self.exp_a = torch.exp(self.params[6])


    def rays_for_pixels(self, u, v):
        '''Compute rays for a batch of pixels.'''

        batch_size = u.shape[0]
        homogeneous_coordinates = torch.ones(batch_size, 3, 1, device='cuda')
        homogeneous_coordinates[:, 0, 0] = u
        homogeneous_coordinates[:, 1, 0] = v

        # Compute ray vectors in camera coordinates, then rotate to world coordinates
        camera_coords = torch.matmul(self.K_inverse, homogeneous_coordinates)
        ray = torch.matmul(self.rotation_matrix, camera_coords)[:,:,0]

        # Normalize the ray vectors to be length 1 (since really we want the direction of the rays)
        with torch.no_grad():
            ray_length_inverse = 1.0 / torch.norm(ray, dim=1).reshape(batch_size,1).expand(batch_size,3)

        normalized_ray = ray * ray_length_inverse

        # We just copy the translation matrix for each ray in the batch because they all originate 
        # from the same point (the camera center)
        return normalized_ray, self.translation_matrix.reshape(1,3).expand(batch_size,3)


def weighted_samples(bin_positions, bin_weights, num_samples):
    """
    Perform hierarchical sampling of bins using inverse transform sampling.

    Parameters:
    bin_positions: The positions of the bins.
    bin_weights: The weights of the bins.
    num_samples: The number of samples to draw.

    Returns:
    samples: The sampled positions.
    """

    # Add a small constant to weights to avoid division by zero
    bin_weights = bin_weights + 1e-5

    # Compute the probability density function (pdf) from the weights
    pdf = bin_weights * torch.reciprocal(torch.sum(bin_weights, -1, keepdim=True))

    # Compute the cumulative distribution function (cdf) from the pdf
    cdf = torch.cumsum(pdf, -1)

    # Append a zero at the beginning of the cdf
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    # Generate random numbers for sampling
    random_numbers = torch.rand(list(cdf.shape[:-1]) + [num_samples]).cuda()

    # Flatten the random numbers tensor for the searchsorted operation
    random_numbers = random_numbers.contiguous()

    # Find the indices where the random numbers would be inserted to maintain the order of the cdf
    indices = torch.searchsorted(cdf, random_numbers, right=True)

    # Clamp the indices to the valid range
    indices_below = torch.max(torch.zeros_like(indices-1), indices-1)
    indices_above = torch.min((cdf.shape[-1]-1) * torch.ones_like(indices), indices)

    # Stack the indices
    indices_grouped = torch.stack([indices_below, indices_above], -1)

    # Expand the cdf and bin_positions tensors to match the shape of indices_grouped
    matched_shape = [indices_grouped.shape[0], indices_grouped.shape[1], cdf.shape[-1]]
    cdf_grouped = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_grouped)
    bin_positions_grouped = torch.gather(bin_positions.unsqueeze(1).expand(matched_shape), 2, indices_grouped)

    # Compute the interpolation weights
    denominator = (cdf_grouped[...,1]-cdf_grouped[...,0])
    denominator = torch.where(denominator<1e-5, torch.ones_like(denominator), denominator)
    interpolation_weights = (random_numbers-cdf_grouped[...,0])/denominator

    # Interpolate the samples
    samples = bin_positions_grouped[...,0] + interpolation_weights * (bin_positions_grouped[...,1]-bin_positions_grouped[...,0])

    # Concatenate the samples with the bin_positions and sort them
    samples_concatenated, _ = torch.sort(torch.cat([samples, bin_positions], -1), dim=-1)

    return samples_concatenated


class Mapper():
    def __init__(self):
        self.model = IMAP().cuda()
        self.model_tracking = IMAP().cuda()
        self.cameras = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.005)
        self.render_id=0

    def freeze_model_for_tracking(self):
        self.model_tracking.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def add_camera(self, rgb_filename, depth_filename, 
                   position_x, position_y, position_z, 
                   rotation_x, rotation_y, rotation_z, 
                   light_scale, light_offset):
        rgb = cv2.imread(rgb_filename, cv2.IMREAD_COLOR)
        depth = cv2.imread(depth_filename, cv2.IMREAD_ANYDEPTH)
        camera = Camera(rgb, depth, 
                        position_x, position_y, position_z, 
                        rotation_x, rotation_y, rotation_z, 
                        light_scale, light_offset)
        self.cameras.append(camera)

    '''
    def render_marching_cube(self, voxel_size=64, threshold=30.0):
        with torch.no_grad():
            vs = 2.4/voxel_size
            t = np.linspace(-1.2, 1.2, voxel_size+1)
            query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
            sampling = torch.from_numpy(query_pts.reshape([-1,3])).cuda()
            print(sampling.cpu())
            out = self.model(sampling)
            sigma=out[:,3].detach().cpu().numpy().reshape(voxel_size+1,voxel_size+1,voxel_size+1)
            print(np.min(sigma), np.max(sigma))
            vertices, triangles = mcubes.marching_cubes(sigma, threshold)
            if vertices.shape[0]==0:
                return np.zeros((3,3), np.float32), np.zeros((3,3), np.float32)
            print(vertices.shape)
            color_sampling = torch.from_numpy(np.stack(vertices, -1).astype(np.float32).reshape([-1,3]))*vs-1.2

            out = self.model(color_sampling.cuda())
            colors = out[:,:3].detach().cpu().numpy()

            vt=vertices[triangles.flatten()]*vs
            cl = colors[triangles.flatten()]
            return vt,cl
    '''

    def volume_render(self, distances, sigmas):
        """
        Perform volume rendering.

        Parameters:
        distances: The distances to the sampled points along the rays.
        sigmas: The density values at the sampled points.

        Returns:
        depth: The rendered depth map.
        intensity: The rendered intensity map.
        depth_variance: The variance of the rendered depth map.
        """

        max_distance = 1.5
        batch_size = distances.shape[0]
        num_steps = distances.shape[1]

        # Compute the step sizes along the rays
        step_sizes = distances[:, 1:] - distances[:, :-1]

        # Compute the opacity of each step
        opacity = 1 - torch.exp(-sigmas[:, :-1, 3] * step_sizes)

        # Compute the accumulated opacity along each ray
        accumulated_opacity = torch.cumprod(1 - opacity, dim=1)

        # Compute the weights for each step
        weights = torch.zeros((batch_size, num_steps - 1), device='cuda')
        weights[:, 1:] = opacity[:, 1:] * accumulated_opacity[:, :-1]

        # Compute the depth and intensity maps
        depth_map = torch.sum(weights * distances[:, :-1], dim=1)
        intensity_map = torch.sum(weights.view(batch_size, -1, 1) * sigmas[:, :-1, :3], dim=1)

        # Compute the variance of the depth map
        depth_variance = torch.sum(weights * torch.square(distances[:, :-1] - depth_map.view(batch_size, 1)), dim=1)

        # Add a background depth and intensity for pixels that don't hit any surface
        depth_map += accumulated_opacity[:, -1] * max_distance

        return depth_map, intensity_map, depth_variance


    def render_rays(self, u, v, camera, num_coarse_samples=32, num_fine_samples=12, use_tracking_model=False):
        """
        Render a batch of rays.

        Parameters:
        u, v: The pixel coordinates of the rays.
        camera: The camera parameters.
        num_coarse_samples: The number of coarse samples to take along each ray.
        num_fine_samples: The number of fine samples to take along each ray.
        freeze_model: Whether to freeze the model during rendering.

        Returns:
        depth: The rendered depth map.
        intensity: The rendered intensity map.
        depth_variance: The variance of the rendered depth map.
        """

        if use_tracking_model:
            model = self.model_tracking
        else:
            model = self.model

        batch_size = u.shape[0]

        # Compute the rays
        ray_direction, ray_origin = camera.rays_for_pixels(u, v)

        # First uniformly sample coarsely along each ray
        with torch.no_grad():
            distances_coarse = torch.linspace(0.0001, 1.2, num_coarse_samples, device='cuda') \
                                    .view(1, num_coarse_samples) \
                                    .expand(batch_size, num_coarse_samples)
            rays_coarse = ray_origin.view(batch_size, 1, 3) + \
                          ray_direction.view(batch_size, 1, 3) * distances_coarse.view(batch_size, num_coarse_samples, 1)

            sigmas_coarse = model(rays_coarse.view(-1, 3)).view(batch_size, num_coarse_samples, 4)

            # Compute the weights for the hierarchical sampling
            step_size = distances_coarse[0, 1] - distances_coarse[0, 0]
            opacity = 1 - torch.exp(-sigmas_coarse[:, :, 3] * step_size)[:, 1:]
            accumulated_transparency = 1 - torch.exp(-torch.cumsum(sigmas_coarse[:, :, 3] * step_size, dim=1))[:, :-1]
            weights = opacity * accumulated_transparency

            # Now perform hierarchical sampling biased towards regions with high opacity
            distances_fine = weighted_samples(distances_coarse, weights, num_fine_samples)

        total_samples = num_coarse_samples + num_fine_samples

        # Compute the fine sample locations along each ray and get their densities from the model
        rays_fine = ray_origin.view(batch_size, 1, 3) + \
                    ray_direction.view(batch_size, 1, 3) * distances_fine.view(batch_size, total_samples, 1)
        sigmas_fine = model(rays_fine.view(-1, 3)).view(batch_size, total_samples, 4)

        # Perform volume rendering with the fine samples
        depth, intensity, depth_variance = self.volume_render(distances_fine, sigmas_fine)

        # Adjust the intensity based on the camera parameters
        intensity = camera.exp_a * intensity + camera.params[7]

        return depth, intensity, depth_variance


    def render_image(self, camera):
        """
        Render an image from a camera view.

        Parameters:
        camera: The camera parameters.

        Returns:
        rgb_image: The rendered RGB image.
        depth_image: The rendered depth map.
        """

        with torch.no_grad():
            height, width = camera.image_size
            depth_map = torch.zeros((height, width), device='cuda')
            rgb_image = torch.zeros((height, width, 3), device='cuda')

            # Render the image in chunks to save memory
            vertical_chunk_size = 40
            for v in range(0, height, vertical_chunk_size):
                vertical_indices = torch.arange(v, min(v + vertical_chunk_size, height)).view(-1, 1).expand(-1, width).reshape(-1).cuda()
                horizontal_indices = torch.arange(width).view(1, -1).expand(vertical_chunk_size, -1).reshape(-1).cuda()

                # Render a chunk of rays
                depth_chunk, rgb_chunk, _ = self.render_rays(horizontal_indices, vertical_indices, camera)

                # Update the depth map and the RGB image with the rendered chunk
                depth_map[v:v + vertical_chunk_size, :] = depth_chunk.view(-1, width)
                rgb_image[v:v + vertical_chunk_size, :, :] = rgb_chunk.view(-1, width, 3)

            # Convert the depth map and the RGB image to numpy arrays
            rgb_image = torch.clamp(rgb_image * 255, 0, 255).detach().cpu().numpy().astype(np.uint8)
            depth_map = torch.clamp(depth_map * 50000 / 256, 0, 255).detach().cpu().numpy().astype(np.uint8)

            return rgb_image, depth_map


    def render_preview_image(self, camera, label, scale_factor=5):
        """
        Render a small preview image from a camera view.

        Parameters:
        camera: The camera parameters.
        label: The label for the preview image.

        Returns:
        None
        """

        with torch.no_grad():
            # Update the camera transform
            camera.update_transform()

            # Compute the size of the preview image
            full_height, full_width = camera.image_size
            height = int(full_height / scale_factor)
            width = int(full_width / scale_factor)

            # Compute the pixel coordinates for the preview image
            vertical_indices = (scale_factor * torch.arange(height)) \
                               .view(-1, 1).expand(-1, width).reshape(-1).cuda()
            horizontal_indices = (scale_factor * torch.arange(width)) \
                                 .view(1, -1).expand(height, -1).reshape(-1).cuda()

            # Render the preview image
            depth_map, rgb_image, _ = self.render_rays(horizontal_indices, vertical_indices, camera)
            depth_map = depth_map.view(-1, width)
            rgb_image = rgb_image.view(-1, width, 3)

            # Convert the depth map and the RGB image to numpy arrays
            rgb_image = torch.clamp(rgb_image * 255, 0, 255).detach().cpu().numpy().astype(np.uint8)
            depth_map = torch.clamp(depth_map * 50000 / 256, 0, 255).detach().cpu().numpy().astype(np.uint8)

            # Get the ground truth RGB image and depth map
            rgb_image_gt = torch.clamp(camera.rgb_image * 255, 0, 255).detach().cpu().numpy().astype(np.uint8)
            depth_map_gt = torch.clamp(camera.depth_image * 50000 / 256, 0, 255).detach().cpu().numpy().astype(np.uint8)

            # Concatenate the rendered and ground truth images for comparison
            rgb_image_preview = cv2.hconcat([cv2.resize(rgb_image, (full_width, full_height)), rgb_image_gt])
            depth_map_preview = cv2.cvtColor(cv2.hconcat([cv2.resize(depth_map, (full_width, full_height)), 
                                                          depth_map_gt]), 
                                             cv2.COLOR_GRAY2RGB)
            preview_image = cv2.vconcat([rgb_image_preview, depth_map_preview])

            # Save the preview image
            cv2.imwrite("render/{}_{:04}.png".format(label, self.render_id), preview_image)
            self.render_id += 1

            # Display the preview image
            cv2.imshow("{}_rgb".format(label), preview_image)
            cv2.waitKey(1)


    def update_map(self, batch_size=200, active_sampling=True):
        """
        Perform mapping with a batch of rays.

        Parameters:
        batch_size: The size of the batch of rays.
        active_sampling: Whether to use active sampling.

        Returns:
        None
        """

        # Select a set of images/cameras to use for the mapping
        if len(self.cameras) < 5:
            camera_ids = np.arange(len(self.cameras))
        else:
            # Select 5 random cameras, but make sure that the most recent two images are included
            camera_ids = np.random.randint(0, len(self.cameras) - 2, 5)
            camera_ids[3] = len(self.cameras) - 1
            camera_ids[4] = len(self.cameras) - 2

        # Perform mapping for each selected camera
        for camera_id in camera_ids:
            # Reset the gradients
            self.optimizer.zero_grad()
            camera = self.cameras[camera_id]
            camera.optimizer.zero_grad()

            # update the camera transform 
            camera.update_transform()

            # compute the size of the image
            height, width = camera.image_size

            # compute the pixel coordinates for the batch of rays
            if active_sampling:
                # use active sampling over an 8x8 grid to select the pixel coordinates
                with torch.no_grad():
                    sub_height = int(height / 8)
                    sub_width = int(width / 8)
                    u_list = []
                    v_list = []

                    # Determine the number of samples for each grid cell
                    # and sample the pixel coordinates
                    num_samples = torch.zeros(64, dtype=torch.int32, device='cuda')
                    for i in range(64):
                        num_samples[i] = int(batch_size * camera.grid_sampling_probs[i])

                        if num_samples[i] < 1:
                            num_samples[i] = 1

                        u_list.append((torch.rand(num_samples[i]) * (sub_width - 1)).to(torch.int16).cuda() + (i % 8) * sub_width)
                        v_list.append((torch.rand(num_samples[i]) * (sub_height - 1)).to(torch.int16).cuda() + int(i / 8) * sub_height)

                    u_coordinates = torch.cat(u_list)
                    v_coordinates = torch.cat(v_list)
            else:
                # Use random sampling to select the pixel coordinates
                u_coordinates = (torch.rand(batch_size) * (width - 1)).to(torch.int16).cuda()
                v_coordinates = (torch.rand(batch_size) * (height - 1)).to(torch.int16).cuda()

            # Render the batch of rays
            depth, rgb, depth_variance = self.render_rays(u_coordinates, v_coordinates, camera)

            # Get the ground truth depth and RGB values
            depth_gt = torch.cat([camera.depth_image[v, u].unsqueeze(0) for u, v in zip(u_coordinates, v_coordinates)])
            rgb_gt = torch.cat([camera.rgb_image[v, u, :].unsqueeze(0) for u, v in zip(u_coordinates, v_coordinates)])

            # Ignore the depth values for pixels that don't hit any surface
            depth[depth_gt == 0] = 0

            # Compute the inverse variance
            with torch.no_grad():
                inverse_variance = torch.reciprocal(torch.sqrt(depth_variance))
                inverse_variance[inverse_variance.isinf()] = 1
                inverse_variance[inverse_variance.isnan()] = 1

            # Compute the loss for the depth and the RGB values
            depth_loss = torch.mean(torch.abs(depth - depth_gt) * inverse_variance)
            rgb_loss = 5 * torch.mean(torch.abs(rgb - rgb_gt))
            total_loss = depth_loss + rgb_loss

            # Backpropagate the loss and update the parameters
            total_loss.backward()
            self.optimizer.step()

            if camera_id > 0:
                self.cameras[camera_id].optimizer.step()

            # Update the active sampling probabilities
            if active_sampling:
                with torch.no_grad():
                    # Compute the error for each sample
                    error = torch.abs(depth - depth_gt) + torch.sum(torch.abs(rgb - rgb_gt), dim=1)

                    # Compute the mean error for each grid cell
                    num_samples_cumsum = torch.cumsum(num_samples, dim=0)
                    active_sampling_probabilities = torch.zeros(64, device='cuda')
                    active_sampling_probabilities[0] = torch.mean(error[:num_samples_cumsum[0]])
                    for i in range(1, 64):
                        active_sampling_probabilities[i] = torch.mean(error[num_samples_cumsum[i - 1]:num_samples_cumsum[i]])

                    # Normalize the active sampling probabilities
                    active_sampling_probabilities_sum = torch.sum(active_sampling_probabilities)
                    self.cameras[camera_id].grid_sampling_probs = active_sampling_probabilities / active_sampling_probabilities_sum


    def track(self, camera, batch_size=200, n_iters=20):
        """
        Perform tracking for a camera view.

        Parameters:
        camera: The camera parameters.
        batch_size: The size of the batch of rays.

        Returns:
        p: The proportion of depth values that are close to the ground truth.
        """

        # Update the model for tracking
        self.freeze_model_for_tracking()

        for _ in range(n_iters):
            # Reset the gradients
            camera.optimizer.zero_grad()

            # Update the camera transform
            camera.update_transform()

            # Compute the size of the image
            height, width = camera.image_size

            # Compute the pixel coordinates for a randomly sampled batch of rays
            u_coordinates = (torch.rand(batch_size) * (width - 1)).to(torch.int16).cuda()
            v_coordinates = (torch.rand(batch_size) * (height - 1)).to(torch.int16).cuda()

            # Render the batch of rays
            depth, rgb, depth_variance = self.render_rays(u_coordinates, v_coordinates, camera, use_tracking_model=True)

            # Get the ground truth depth and RGB values
            depth_gt = torch.cat([camera.depth_image[v, u].unsqueeze(0) for u, v in zip(u_coordinates, v_coordinates)])
            rgb_gt = torch.cat([camera.rgb_image[v, u, :].unsqueeze(0) for u, v in zip(u_coordinates, v_coordinates)])

            # Ignore the depth values for pixels that don't hit any surface
            depth[depth_gt == 0] = 0

            # Compute the inverse variance
            with torch.no_grad():
                inverse_variance = torch.reciprocal(torch.sqrt(depth_variance))
                inverse_variance[inverse_variance.isinf()] = 1
                inverse_variance[inverse_variance.isnan()] = 1

            # The depth loss is inversely weighted by the depth variance
            depth_loss = torch.mean(torch.abs(depth - depth_gt) * inverse_variance)

            # The RGB loss is weighted as more important by a factor of 5
            rgb_loss = 5 * torch.mean(torch.abs(rgb - rgb_gt))

            total_loss = depth_loss + rgb_loss

            # Backpropagate the loss and update the parameters
            total_loss.backward()
            camera.optimizer.step()

            # Compute the proportion of depth values that are close to the ground truth
            p = float(torch.sum(((torch.abs(depth - depth_gt) * torch.reciprocal(depth_gt + 1e-12)) < 0.1).int()).cpu().item()) / batch_size

            # If the proportion is high enough, short circuit stop the tracking
            if p > 0.8:
                break

        print("Tracking: P=", p)

        return p