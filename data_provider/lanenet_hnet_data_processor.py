#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 下午3:33
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet中的HNet训练数据流
"""
import os.path as ops
import json
import cv2
import numpy as np

try:
    from cv2 import cv2
except ImportError:
    pass


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file):
        """

        :param dataset_info_file: json文件列表
        """
        self._label_image_path, self._label_gt_pts = self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """
        从json标注文件中获取标注样本信息
        :param dataset_info_file:
        :return:
        """
        label_image_path = []
        label_gt_pts = []

        for json_file_path in dataset_info_file:
            assert ops.exists(json_file_path), '{:s} not exist'.format(json_file_path)

            src_dir = ops.split(json_file_path)[0]

            with open(json_file_path, 'r') as file:
                for line in file:
                    info_dict = json.loads(line)

                    image_dir = ops.split(info_dict['raw_file'])[0]
                    image_dir_split = image_dir.split('/')[1:]
                    image_dir_split.append(ops.split(info_dict['raw_file'])[1])
                    image_path = ops.join(src_dir, info_dict['raw_file'])
                    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

                    label_image_path.append(image_path)

                    h_samples = info_dict['h_samples']
                    lanes = info_dict['lanes']

                    lane_pts = []
                    for lane in lanes:
                        assert len(h_samples) == len(lane)
                        for index in range(len(lane)):
                            if lane[index] == -2:
                                continue
                            else:
                                ptx = lane[index]
                                pty = h_samples[index]
                                ptz = 1
                                lane_pts.append([ptx, pty, ptz])
                        if not lane_pts:
                            continue
                        if len(lane_pts) <= 3:
                            continue
                    label_gt_pts.append(lane_pts)

        return np.array(label_image_path), np.array(label_gt_pts)

    def _random_dataset(self):
        """

        :return:
        """
        assert self._label_image_path.shape[0] == self._label_gt_pts.shape[0]

        random_idx = np.random.permutation(self._label_image_path.shape[0])
        self._label_image_path = self._label_image_path[random_idx]
        self._label_gt_pts = self._label_gt_pts[random_idx]

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert self._label_gt_pts.shape[0] == self._label_image_path.shape[0]

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > self._label_image_path.shape[0]:
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._label_image_path[idx_start:idx_end]
            gt_pts_list = self._label_gt_pts[idx_start:idx_end]

            gt_imgs = []

            for gt_img_path in gt_img_list:
                gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR))

            self._next_batch_loop_count += 1
            return gt_imgs, gt_pts_list


if __name__ == '__main__':
    import glob
    json_file_list = glob.glob('{:s}/*.json'.format('/media/baidu/Data/Semantic_Segmentation'
                                                    '/TUSimple_Lane_Detection/training'))
    json_file_list = [tmp for tmp in json_file_list if 'test' not in tmp]
    val = DataSet(json_file_list)
    a1, a2 = val.next_batch(2)
    print(a1)
    print(a2)
    src_image = cv2.imread(a1[0], cv2.IMREAD_COLOR)
    image = np.zeros(shape=[src_image.shape[0], src_image.shape[1]], dtype=np.uint8)
    for pt in a2[0]:
        ptx = pt[0]
        pty = pt[1]
        image[pty, ptx] = 255

    import matplotlib.pyplot as plt
    plt.imshow(image, cmap='gray')
    plt.show()
