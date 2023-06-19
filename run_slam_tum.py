import time
import torch
import sys
import glob
import os
import csv
import cv2
import threading
from model import Camera, Mapper


def read_rgbd_files(folder_path=r"/home/rosejn/datasets/tum_rgbd/rgbd_dataset_freiburg1_teddy/"):
    csv_file = open(folder_path + "rgb.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    next(f)
    next(f)
    next(f)
    rgb_filenames = []
    for row in f:
        rgb_filenames.append("{}{}".format(folder_path, row[1]))
    csv_file = open(folder_path + "depth.txt", "r")
    f = csv.reader(csv_file, delimiter=" ")
    next(f)
    next(f)
    next(f)
    depth_filenames = []
    for row in f:
        depth_filenames.append("{}{}".format(folder_path, row[1]))

    return rgb_filenames, depth_filenames


def mapping_loop(mapper):
    while True:
        mapper.update_map()
        time.sleep(0.1)
        print("update map")


def main():
    mapper = Mapper()

    if 2<= len(sys.argv):
        rgb_filenames, depth_filenames = read_rgbd_files(sys.argv[1])
    else:
        rgb_filenames, depth_filenames = read_rgbd_files()

    frame_length = min(len(rgb_filenames), len(depth_filenames))

    mapper.add_camera(rgb_filenames[0], depth_filenames[0],
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0)

    fixed_camera = Camera(cv2.imread(rgb_filenames[0], cv2.IMREAD_COLOR),
                          cv2.imread(depth_filenames[0], cv2.IMREAD_ANYDEPTH),
                          0.0, 0.0, 0.0,
                          1e-8, 1e-8, 1e-8,
                          0.0, 0.0)

    tracking_camera = Camera(cv2.imread(rgb_filenames[0], cv2.IMREAD_COLOR),
                             cv2.imread(depth_filenames[0], cv2.IMREAD_ANYDEPTH),
                             0.0, 0.0, 0.0,
                             1e-8, 1e-8, 1e-8,
                             0.0, 0.0)

    # Initialize the map with 200 iterations
    print("Initializing map")
    for i in range(200):
        mapper.update_map(batch_size=200, active_sampling=False)
    print("Map initialized")

    # For calc kinematics for camera motion
    last_pose = tracking_camera.params
    camera_vel = torch.tensor([0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 
                               0.0, 0.0]).detach().cuda().requires_grad_(True)

    last_keyframe = 0

    mapping_thread = threading.Thread(target=mapping_loop, args=(mapper,))
    mapping_thread.start()

    for cur_frame in range(1, frame_length):
        tracking_camera.params.data += camera_vel
        tracking_camera.set_images(cv2.imread(rgb_filenames[cur_frame], cv2.IMREAD_COLOR),
                                   cv2.imread(depth_filenames[cur_frame], cv2.IMREAD_ANYDEPTH))

        pe = mapper.track(tracking_camera)
        camera_vel = 0.2 * camera_vel + 0.8 * (tracking_camera.params - last_pose)
        last_pose = tracking_camera.params

        # Add keyframe when the tracking error grows over a threshold
        if pe < 0.65 and (cur_frame - last_keyframe) > 5:
            p = tracking_camera.params
            mapper.add_camera(rgb_filenames[cur_frame],
                              depth_filenames[cur_frame],
                              last_pose[3], last_pose[4], last_pose[5],
                              last_pose[0], last_pose[1], last_pose[2],
                              last_pose[6], last_pose[7])

            print("Adding keyframe")
            print(last_pose.cpu())
            last_keyframe = cur_frame

        # Render from tracking camera
        mapper.render_preview_image(tracking_camera, "view")

        # Render from fixed camera
        #mapper.render_small(fixed_camera, "fixed_camera")

    mapping_thread.join()

main()