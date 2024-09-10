# -*- coding: utf-8 -*-
#        Data: 2024-09-09 17:54
#     Project: LoFTR
#   File Name: read_dense.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

#!/usr/bin/env python

# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import argparse
import numpy as np
import os
import struct
import h5py
from tqdm import tqdm
import cv2


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(
            endian_character + format_char_sequence, *data_list
        )
        fid.write(byte_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--depth_map_dir", help="path to depth map dir", type=str,
    )
    parser.add_argument(
        "-n", "--normal_map_dir", help="path to normal map dir", type=str,
    )
    parser.add_argument(
        "-o", "--output_dir", help="path to output dir for h5 file", type=str,
    )
    parser.add_argument(
        "--min_depth_percentile",
        help="minimum visualization depth percentile",  # 最小可视化深度百分位数
        type=float,
        default=5,
    )
    parser.add_argument(
        "--max_depth_percentile",
        help="maximum visualization depth percentile",  # 最大可视化深度百分位数
        type=float,
        default=95,
    )
    args = parser.parse_args()
    return args


def depth_to_h5(depth_map, file_name, output_dir):
    """将深度图转换为h5文件并保存到指定目录"""
    depth_h5_dir = os.path.join(os.path.dirname(output_dir), "depth")
    os.makedirs(depth_h5_dir, exist_ok=True)

    # 获取文件名
    h5_path = os.path.join(depth_h5_dir, file_name.split(".")[0] + '.h5')

    # depth = depth_map / 100  # 如果深度值过大，需将数值缩放到0-100的范围内，直接除以10^n即可，需要根据实际情况确定
    with h5py.File(h5_path, "w") as f:
        dest = f.create_dataset("depth", data=depth_map, compression='gzip', compression_opts=9)
        dest.attrs["description"] = "Depth map of the image"


def depth_to_image(depth_map, file_name, output_dir, min_depth, max_depth):
    """将深度图转换为图像并保存到指定目录"""
    depth_image_dir = os.path.join(os.path.dirname(output_dir), "depth_image")
    os.makedirs(depth_image_dir, exist_ok=True)
    depth_map_image = (depth_map - min_depth) / (max_depth - min_depth) * 255.  # 将深度值转换为像素值进行可视化
    depth_map_image = cv2.applyColorMap(depth_map_image.astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(depth_image_dir, file_name.split(".")[0] + ".png"), depth_map_image)


def show_map(depth_map, normal_map):
    import pylab as plt

    # Visualize the depth map.
    plt.figure()
    plt.imshow(depth_map)
    plt.title("depth map")

    # Visualize the normal map.
    plt.figure()
    plt.imshow(normal_map)
    plt.title("normal map")

    plt.show()


def main():
    args = parse_args()
    # args.depth_map_dir = r"E:\Pycharm\3D_Reconstruct\datasets\forest\forest_new\dense\stereo\depth_maps"
    # args.normal_map_dir = r"E:\Pycharm\3D_Reconstruct\datasets\forest\forest_new\dense\stereo\normal_maps"
    # args.output_dir = r"E:\Pycharm\3D_Reconstruct\datasets\forest\forest_new\depth"

    if args.min_depth_percentile > args.max_depth_percentile:
        raise ValueError(
            "min_depth_percentile should be less than or equal "
            "to the max_depth_percentile."
        )
    image_name_list = [file_name.rsplit(".", 2)[0] for file_name in os.listdir(args.depth_map_dir)]
    depth_path_list = [os.path.join(args.depth_map_dir, file_name+".photometric.bin") for file_name in image_name_list]
    normal_path_list = [os.path.join(args.normal_map_dir, file_name+".photometric.bin") for file_name in image_name_list]
    for file_name, depth_map_path, normal_map_path in tqdm(zip(image_name_list, depth_path_list, normal_path_list), total=len(image_name_list), desc="Convert Depth to h5"):
        # Read depth and normal maps corresponding to the same image.
        if not os.path.exists(depth_map_path):
            raise FileNotFoundError("File not found: {}".format(depth_map_path))

        if not os.path.exists(normal_map_path):
            raise FileNotFoundError("File not found: {}".format(normal_map_path))

        depth_map = read_array(depth_map_path)
        # normal_map = read_array(normal_map_path)

        min_depth, max_depth = np.percentile(
            depth_map, [args.min_depth_percentile, args.max_depth_percentile]
        )
        depth_map[depth_map < min_depth] = min_depth
        depth_map[depth_map > max_depth] = max_depth
        # 根据深度图中的深度值分布来确定深度值范围
        # 通过设置最小和最大深度值，可以去除这些异常值，从而提高结果的准确性

        # save depth map image
        depth_to_image(depth_map, file_name, args.output_dir, min_depth, max_depth)
        # save depth map h5
        depth_to_h5(depth_map, file_name, args.output_dir)

        # show depth and noraml maps
        # show_map(depth_map, normal_map)



if __name__ == "__main__":
    main()