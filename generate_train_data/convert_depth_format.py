# -*- coding: utf-8 -*-
#        Data: 2024-09-06 12:08
#     Project: LoFTR
#   File Name: depth_evaluate.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description: 2.使用Depth-Anything仓库的预训练模型对照片进行深度估计，并将结果转为h5格式
import os
import h5py
import cv2
from tqdm import tqdm
import argparse


def image_to_h5(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件名
    image_name = os.path.basename(image_path).split('.')[0]
    h5_path = os.path.join(output_dir, image_name + '.h5')

    # 读取image格式深度图
    depth = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    depth = depth / 100  # 对深度值进行缩放
    with h5py.File(h5_path, "w") as f:
        # dest = f.create_dataset("depth", data=depth, compression='gzip', compression_opts=9)  # small file but slow
        dest = f.create_dataset("depth", data=depth)  # big file but fast
        dest.attrs["description"] = "Depth map of the image"

    # depth = h5py.File(h5_path, "r")['depth']
    # print(depth.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Path to image")
    parser.add_argument("--output_dir", required=True, help="Path to output directory")
    args = parser.parse_args()

    all_image_path_list = [os.path.join(args.image_dir, image_name) for image_name in os.listdir(args.image_dir)]
    for image_path in tqdm(all_image_path_list, total=len(all_image_path_list), desc="Trans to h5"):
        image_to_h5(image_path, args.output_dir)