# -*- coding: utf-8 -*-
#        Data: 2024-09-06 12:08
#     Project: LoFTR
#   File Name: depth_evaluate.py
#      Author: KangPeilun
#       Email: 374774222@qq.com
# Description: 3.将数据转换成MegaDepth数据集格式

import argparse
import numpy as np
import os
from tqdm import tqdm
import random
import shutil

parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

parser.add_argument(
    '--base_path', type=str, required=True,
    help='path to MegaDepth'
)
parser.add_argument(
    '--scene_id', type=str, required=True,
    help='scene ID'
)
parser.add_argument(
    '--output_path', type=str, required=True,
    help='path to the output directory'
)


args = parser.parse_args()

base_path = args.base_path
# Remove the trailing / if need be.
if base_path[-1] in ['/', '\\']:
    base_path = base_path[: - 1]
scene_id = args.scene_id


undistorted_sparse_path = os.path.join(  # sfm文件夹
    base_path, "sparse/0"
)
if not os.path.exists(undistorted_sparse_path):
    exit()

depths_path = os.path.join(
    base_path, "depth"
)
if not os.path.exists(depths_path):
    exit()

images_path = os.path.join(
    base_path, 'images'
)
if not os.path.exists(images_path):
    exit()

# Process cameras.txt
with open(os.path.join(undistorted_sparse_path, 'cameras.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header

camera_intrinsics = {}
for camera in raw:
    camera = camera.split(' ')
    camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]]

# Process points3D.txt
with open(os.path.join(undistorted_sparse_path, 'points3D.txt'), 'r') as f:
    raw = f.readlines()[3 :]  # skip the header

points3D = {}
for point3D in tqdm(raw, total=len(raw), desc="Process points3D.txt"):
    point3D = point3D.split(' ')
    points3D[int(point3D[0])] = np.array([
        float(point3D[1]), float(point3D[2]), float(point3D[3])
    ])
    
# Process images.txt
with open(os.path.join(undistorted_sparse_path, 'images.txt'), 'r') as f:
    raw = f.readlines()[4:]  # skip the header

image_id_to_idx = {}
image_names = []
raw_pose = []
camera = []
points3D_id_to_2D = []
n_points3D = []
for idx, (image, points) in tqdm(enumerate(zip(raw[:: 2], raw[1 :: 2])), total=len(raw[:: 2]), desc="Process images.txt"):
    image = image.split(' ')
    points = points.split(' ')

    image_id_to_idx[int(image[0])] = idx

    image_name = image[-1].strip('\n')
    image_names.append(image_name)

    raw_pose.append([float(elem) for elem in image[1: -2]])
    camera.append(int(image[-2]))
    current_points3D_id_to_2D = {}
    for x, y, point3D_id in zip(points[:: 3], points[1 :: 3], points[2 :: 3]):
        if int(point3D_id) == -1:
            continue
        current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
    points3D_id_to_2D.append(current_points3D_id_to_2D)
    n_points3D.append(len(current_points3D_id_to_2D))
n_images = len(image_names)

# Image and depthmaps paths
image_paths = []
depth_paths = []
for image_name in tqdm(image_names, total=len(image_names), desc="Image and depthmaps paths"):
    image_path = os.path.join(images_path, image_name)
   
    # Path to the depth file
    depth_path = os.path.join(
        depths_path, '%s.h5' % os.path.splitext(image_name)[0]
    )
    
    if os.path.exists(depth_path):
        # Check if depth map or background / foreground mask
        file_size = os.stat(depth_path).st_size
        # Rough estimate - 75KB might work as well
        if file_size < 100 * 1024:
            depth_paths.append(None)
            image_paths.append(None)
        else:
            depth_paths.append(depth_path[len(base_path) + 1:])
            image_paths.append(image_path[len(base_path) + 1:])
    else:
        depth_paths.append(None)
        image_paths.append(None)

# Camera configuration
intrinsics = []
poses = []
principal_axis = []
points3D_id_to_ndepth = []
for idx, image_name in tqdm(enumerate(image_names), total=len(image_names), desc="Process Camera configuration"):
    if image_paths[idx] is None:
        intrinsics.append(None)
        poses.append(None)
        principal_axis.append([0, 0, 0])
        points3D_id_to_ndepth.append({})
        continue
    image_intrinsics = camera_intrinsics[camera[idx]]
    K = np.zeros([3, 3])
    K[0, 0] = image_intrinsics[2]
    K[0, 2] = image_intrinsics[4]
    K[1, 1] = image_intrinsics[3]
    K[1, 2] = image_intrinsics[5]
    K[2, 2] = 1
    intrinsics.append(K)

    image_pose = raw_pose[idx]
    qvec = image_pose[: 4]
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([
        [
            1 - 2 * y * y - 2 * z * z,
            2 * x * y - 2 * z * w,
            2 * x * z + 2 * y * w
        ],
        [
            2 * x * y + 2 * z * w,
            1 - 2 * x * x - 2 * z * z,
            2 * y * z - 2 * x * w
        ],
        [
            2 * x * z - 2 * y * w,
            2 * y * z + 2 * x * w,
            1 - 2 * x * x - 2 * y * y
        ]
    ])
    principal_axis.append(R[2, :])
    t = image_pose[4 : 7]
    # World-to-Camera pose
    current_pose = np.zeros([4, 4])
    current_pose[: 3, : 3] = R
    current_pose[: 3, 3] = t
    current_pose[3, 3] = 1
    # Camera-to-World pose
    # pose = np.zeros([4, 4])
    # pose[: 3, : 3] = np.transpose(R)
    # pose[: 3, 3] = -np.matmul(np.transpose(R), t)
    # pose[3, 3] = 1
    poses.append(current_pose)
    
    current_points3D_id_to_ndepth = {}
    for point3D_id in points3D_id_to_2D[idx].keys():
        p3d = points3D[point3D_id]
        current_points3D_id_to_ndepth[point3D_id] = (np.dot(R[2, :], p3d) + t[2]) / (.5 * (K[0, 0] + K[1, 1])) 
    points3D_id_to_ndepth.append(current_points3D_id_to_ndepth)
principal_axis = np.array(principal_axis)
angles = np.rad2deg(np.arccos(
    np.clip(
        np.dot(principal_axis, np.transpose(principal_axis)),
        -1, 1
    )
))

# Compute overlap score 计算重叠分数
overlap_matrix = np.full([n_images, n_images], -1.)  # 得到这么多图像中，每对图像的重叠分数，这是一个置信度矩阵
scale_ratio_matrix = np.full([n_images, n_images], -1.)
for idx1 in tqdm(range(n_images), total=n_images, desc="Compute overlap score"):
    if image_paths[idx1] is None or depth_paths[idx1] is None:
        continue
    for idx2 in range(idx1 + 1, n_images):
        if image_paths[idx2] is None or depth_paths[idx2] is None:
            continue
        matches = (
            points3D_id_to_2D[idx1].keys() &
            points3D_id_to_2D[idx2].keys()
        )
        min_num_points3D = min(
            len(points3D_id_to_2D[idx1]), len(points3D_id_to_2D[idx2])
        )
        overlap_matrix[idx1, idx2] = len(matches) / len(points3D_id_to_2D[idx1])  # min_num_points3D
        overlap_matrix[idx2, idx1] = len(matches) / len(points3D_id_to_2D[idx2])  # min_num_points3D
        # if len(matches) == 0:
        #     continue
        # points3D_id_to_ndepth1 = points3D_id_to_ndepth[idx1]
        # points3D_id_to_ndepth2 = points3D_id_to_ndepth[idx2]
        # nd1 = np.array([points3D_id_to_ndepth1[match] for match in matches])
        # nd2 = np.array([points3D_id_to_ndepth2[match] for match in matches])
        # min_scale_ratio = np.min(np.maximum(nd1 / nd2, nd2 / nd1))
        # scale_ratio_matrix[idx1, idx2] = min_scale_ratio
        # scale_ratio_matrix[idx2, idx1] = min_scale_ratio


# get all image Pair
# pair_infos = []
# for image_id in tqdm(range(n_images), total=n_images, desc="Get image pairs"):
#     overlap_score = overlap_matrix[image_id, :]  # 单张图像与各个图像的匹配分数
#     match_pairs = np.where(overlap_score > 0.7)[0]
#     for match_id in match_pairs:
#         pair_infos.append((np.array([image_id, match_id]),  # 匹配对
#                            overlap_score[match_id],      # 该匹配对的重叠分数
#                            None))                         # 占位，使得与MegaDepth数据中的pair_infos数据格式一致，这个值其实在LoFTR中没有用到


# train val test split
# 按照6:2:2的比率划分数据集
random.seed(2024)  # 保证每次划分的结果一样
image_ids = list(range(n_images))  # 一共有多少个数据，每个数据都有一个id
train_sample = random.sample(image_ids, int(n_images * 0.8))  # 采样训练集
val_sample = random.sample(list(set(image_ids) - set(train_sample)), int(n_images * 0.1))  # 采样验证集
test_sample = list(set(image_ids) - set(train_sample) - set(val_sample))  # 采样测试集

train_mask = np.array([True if id in train_sample else False for id in image_ids])  # 获取train val test的mask，方便后面选点
val_mask = np.array([True if id in val_sample else False for id in image_ids])
test_mask = np.array([True if id in test_sample else False for id in image_ids])

# train split
# 将重叠率大于0.7的图像作为匹配的像对
# 一张图像可能对应多个像对
pair_infos = []
for image_id in tqdm(train_sample, total=len(train_sample), desc="Get image pairs train"):
    overlap_score = overlap_matrix[image_id, :]  # 单张图像与各个图像的匹配分数
    match_pairs = np.where(overlap_score > 0.7)[0]
    for match_id in match_pairs:
        if match_id in train_sample:
            pair_infos.append((np.array([image_id, match_id]),  # 匹配对
                               overlap_score[match_id],      # 该匹配对的重叠分数
                               None))                         # 占位，使得与MegaDepth数据中的pair_infos数据格式一致，这个值其实在LoFTR中没有用到

train_image_paths = np.array(image_paths.copy(), dtype=np.object)
train_image_paths[~train_mask] = None
train_depth_paths = np.array(depth_paths.copy(), dtype=np.object)
train_depth_paths[~train_mask] = None
train_intrinsics = np.array(intrinsics.copy(), dtype=np.object)
train_intrinsics[~train_mask] = None
train_poses = np.array(poses.copy(), dtype=np.object)
train_poses[~train_mask] = None

np.savez(
    os.path.join(args.output_path, '%s_train.npz' % scene_id),
    image_paths=train_image_paths,
    depth_paths=train_depth_paths,
    intrinsics=train_intrinsics,
    poses=train_poses,
    pair_infos=pair_infos,
    # overlap_matrix=overlap_matrix,
    # scale_ratio_matrix=scale_ratio_matrix,
    # angles=angles,
    # n_points3D=n_points3D,
    # points3D_id_to_2D=points3D_id_to_2D,
    # points3D_id_to_ndepth=points3D_id_to_ndepth
)

# val split
pair_infos = []
for image_id in tqdm(val_sample, total=len(val_sample), desc="Get image pairs Val"):
    overlap_score = overlap_matrix[image_id, :]  # 单张图像与各个图像的匹配分数
    match_pairs = np.where(overlap_score > 0.7)[0]
    for match_id in match_pairs:
        if match_id in val_sample:
            pair_infos.append((np.array([image_id, match_id]),  # 匹配对
                               overlap_score[match_id],      # 该匹配对的重叠分数
                               None))

val_image_paths = np.array(image_paths.copy(), dtype=np.object)
val_image_paths[~val_mask] = None
val_depth_paths = np.array(depth_paths.copy(), dtype=np.object)
val_depth_paths[~val_mask] = None
val_intrinsics = np.array(intrinsics.copy(), dtype=np.object)
val_intrinsics[~val_mask] = None
val_poses = np.array(poses.copy(), dtype=np.object)
val_poses[~val_mask] = None

np.savez(
    os.path.join(args.output_path, '%s_val.npz' % scene_id),
    image_paths=val_image_paths,
    depth_paths=val_depth_paths,
    intrinsics=val_intrinsics,
    poses=val_poses,
    pair_infos=pair_infos,
)

# test split
pair_infos = []
for image_id in tqdm(test_sample, total=len(test_sample), desc="Get image pairs test"):
    overlap_score = overlap_matrix[image_id, :]  # 单张图像与各个图像的匹配分数
    match_pairs = np.where(overlap_score > 0.7)[0]
    for match_id in match_pairs:
        if match_id in test_sample:
            pair_infos.append((np.array([image_id, match_id]),  # 匹配对
                               overlap_score[match_id],      # 该匹配对的重叠分数
                               None))

test_image_paths = np.array(image_paths.copy(), dtype=np.object)
test_image_paths[~test_mask] = None
test_depth_paths = np.array(depth_paths.copy(), dtype=np.object)
test_depth_paths[~test_mask] = None
test_intrinsics = np.array(intrinsics.copy(), dtype=np.object)
test_intrinsics[~test_mask] = None
test_poses = np.array(poses.copy(), dtype=np.object)
test_poses[~test_mask] = None

np.savez(
    os.path.join(args.output_path, '%s_test.npz' % scene_id),
    image_paths=test_image_paths,
    depth_paths=test_depth_paths,
    intrinsics=test_intrinsics,
    poses=test_poses,
    pair_infos=pair_infos,
)

# copy images and depths
# train
os.makedirs(os.path.join(base_path, "train/images"), exist_ok=True)
os.makedirs(os.path.join(base_path, "train/depth"), exist_ok=True)
for image_path, depth_path in tqdm(zip(train_image_paths, train_depth_paths), total=len(train_image_paths),
                                   desc="Copy train images"):
    if image_path == None: continue
    shutil.copy(os.path.join(base_path, image_path), os.path.join(base_path, "train/images"))
    shutil.copy(os.path.join(base_path, depth_path), os.path.join(base_path, "train/depth"))

# val
os.makedirs(os.path.join(base_path, "val/images"), exist_ok=True)
os.makedirs(os.path.join(base_path, "val/depth"), exist_ok=True)
for image_path, depth_path in tqdm(zip(val_image_paths, val_depth_paths), total=len(train_image_paths),
                                   desc="Copy val images"):
    if image_path == None: continue
    shutil.copy(os.path.join(base_path, image_path), os.path.join(base_path, "val/images"))
    shutil.copy(os.path.join(base_path, depth_path), os.path.join(base_path, "val/depth"))

# test
os.makedirs(os.path.join(base_path, "test/images"), exist_ok=True)
os.makedirs(os.path.join(base_path, "test/depth"), exist_ok=True)
for image_path, depth_path in tqdm(zip(test_image_paths, test_depth_paths), total=len(train_image_paths),
                                   desc="Copy test images"):
    if image_path == None: continue
    shutil.copy(os.path.join(base_path, image_path), os.path.join(base_path, "test/images"))
    shutil.copy(os.path.join(base_path, depth_path), os.path.join(base_path, "test/depth"))

