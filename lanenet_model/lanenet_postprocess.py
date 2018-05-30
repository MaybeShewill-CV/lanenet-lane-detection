#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet模型后处理
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass


class LaneNetPoseProcessor(object):
    """

    """
    def __init__(self):
        """

        """
        pass

    @staticmethod
    def morphological_closing(image, kernel_size=5):
        """

        :param image:
        :param kernel_size:
        :return:
        """
        kernel = np.ones((kernel_size, kernel_size),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=kernel)

    @staticmethod
    def connect_components_analysis(image):
        """

        :param image:
        :return:
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


if __name__ == '__main__':
    processor = LaneNetPoseProcessor()

    image = cv2.imread('test.png', cv2.IMREAD_UNCHANGED)

    ret = processor.morphological_closing(image)
    plt.figure('mor ret')
    plt.imshow(ret)

    ret = processor.connect_components_analysis(ret)
    num_labels = ret[0]
    labels = ret[1]
    stats = ret[2]
    centroids = ret[3]
    plt.figure('connect')
    plt.imshow(labels * 20)

    tan = []

    for stat in stats:
        width = stat[2]
        height = stat[3]
        lt_x = stat[0]
        lt_y = stat[1]
        if lt_x == 0 or lt_y == 0:
            continue
        pt1 = (lt_x, lt_y)
        pt2 = (lt_x + width, lt_y)
        pt3 = (lt_x, lt_y + height)
        pt4 = (lt_x + width, lt_y + height)
        cv2.rectangle(image, pt1, pt4, 255, 3)

        print('tan: {:.5f}'.format(height / width))
        tan.append(height / width)
        # cv2.circle(image, (lt_x, lt_y), 5, 255, 3)

    tan = np.array(tan, np.float32).reshape(-1, 1)
    db = DBSCAN(eps=0.05, min_samples=1).fit(tan)
    db_labels = db.labels_
    unique_labels = np.unique(db_labels)
    unique_labels = [tmp for tmp in unique_labels if tmp != -1]
    print('聚类簇个数为: {:d}'.format(len(unique_labels)))

    num_clusters = len(unique_labels)
    cluster_centers = db.components_

    plt.figure('src')
    plt.imshow(image)

    print('连通域一共有: {:d}'.format(num_labels))
    print(stats)
    print(centroids)

    plt.show()
