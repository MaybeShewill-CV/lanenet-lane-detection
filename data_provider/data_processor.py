#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-31 上午11:21
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : data_processor.py
# @IDE: PyCharm Community Edition
"""
实现训练数据提供类
"""
import os.path as ops

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

        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_list = self._init_dataset(dataset_info_file)
        self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file):
        """

        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                # assert ops.exists(info_tmp[0]) and ops.exists(info_tmp[1])

                gt_img_list.append(info_tmp[0])
                gt_label_list.append(info_tmp[1])

        return gt_img_list, gt_label_list

    def _random_dataset(self):
        """

        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_list.append(self._gt_label_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_list = new_gt_label_list

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_list) == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_list = self._gt_label_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels = []

            for gt_img_path in gt_img_list:
                gt_imgs.append(cv2.imread(gt_img_path, cv2.IMREAD_COLOR))

            for gt_label_path in gt_label_list:
                label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
                label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
                idx = np.where((label_img[:, :, :] == [255, 255, 255]).all(axis=2))
                label_binary[idx] = 1
                gt_labels.append(label_binary)
            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels


if __name__ == '__main__':
    val = DataSet('/home/baidu/DataBase/Semantic_Segmentation/TUSimple_Lane_Detection/training/train.txt')
    a1, a2 = val.next_batch(50)
    b1, b2 = val.next_batch(50)
    c1, c2 = val.next_batch(50)
    dd, d2 = val.next_batch(50)
