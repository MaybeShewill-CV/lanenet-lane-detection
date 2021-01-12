#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def minmax_scale(input_arr):
    if len(input_arr) <= 0:
        return input_arr
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    max_val = max(max_val, 1)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
    return output_arr

def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

def _color_map(idx):
    color_map = np.asarray([
          [120, 120, 120],
	  [180, 120, 120],
	  [6, 230, 230],
	  [80, 50, 50],
	  [4, 200, 3],
	  [120, 120, 80],
	  [140, 140, 140],
	  [204, 5, 255],
	  [230, 230, 230],
	  [4, 250, 7],
	  [224, 5, 255],
	  [235, 255, 7],
	  [150, 5, 61],
	  [120, 120, 70],
	  [8, 255, 51],
	  [255, 6, 82],
	  [143, 255, 140]])
    return color_map[idx]

class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """

        """
    @staticmethod
    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES)
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)
        num_clusters = len(unique_labels)
        cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )

        # dbscan cluster
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats']
        )

        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None:
            return None, None

        lane_coords = []
        for index, label in enumerate(unique_labels.tolist()):
            if label == -1:
                continue
            idx = np.where(db_labels == label)
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
            mask[pix_coord_idx] = _color_map(index)
            lane_coords.append(coord[idx])

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, cfg, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'):
        """

        :param ipm_remap_file_path: ipm generate file path
        """
        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg)
        self._ipm_remap_file_path = ipm_remap_file_path

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x']
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y']

    def _load_remap_matrix(self):
        """

        :return:
        """
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release()

        return ret

    def postprocess2(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=10, source_image=None,
                    data_source='tusimple'):
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 1, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        #morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)
        #connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)
        '''
        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0
        '''
        # get embedding feats and coords
        idxs1 = np.where(binary_seg_result == 1)
        idxs2 = np.where(binary_seg_result == 2)
        idxs3 = np.where(binary_seg_result == 3)
        idxs4 = np.where(binary_seg_result == 4)
        idxs5 = np.where(binary_seg_result == 5)
        idxs6 = np.where(binary_seg_result == 6)
        idxs7 = np.where(binary_seg_result == 7)
        binary_image = source_image.copy()
        binary_image[idxs1] = (0, 255, 255)
        binary_image[idxs2] = (255, 255, 0)
        if len(idxs1[0]) <= 0:
            ret = {
                'binary_image': source_image,
                'instance_image': source_image,
            }
            return ret 
        slices = instance_seg_result[idxs2]
        log.info(slices.shape)
        log.info(type(slices))
        for i in range(3):
            slices[:, i] = minmax_scale(slices[:, i])
        mask_seg = slices[:, (0,1,2)]
        #print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ', mask_seg[:][0])
        instance_image = source_image.copy()
        for i in range(len(idxs2[0])):
            #print(slices[i])
            #'''
            #if ((mask_seg[i][0] > 250 and mask_seg[i][1] < 5 and mask_seg[i][2] > 250)
               #or (mask_seg[i][0] < 5 and mask_seg[i][1] > 250 and mask_seg[i][2] > 250)
               #or (mask_seg[i][0] > 250 and mask_seg[i][1] > 250 and mask_seg[i][2] < 5)
               #or (mask_seg[i][0] < 5 and mask_seg[i][1] < 5 and mask_seg[i][2] > 250)):
               #instance_image[idxs[0][i], idxs[1][i]] = mask_seg[i]
            #'''
            instance_image[idxs2[0][i], idxs2[1][i]] = mask_seg[i]
        #np.set_printoptions(suppress = True, precision = 2, threshold = np.inf)
        #mask_image = np.array(mask_image, np.uint8)
        #print(mask_image)
        instance_image[idxs1] = _color_map(1)
        instance_image[idxs3] = _color_map(3)
        instance_image[idxs4] = _color_map(4)
        instance_image[idxs5] = _color_map(5)
        instance_image[idxs6] = _color_map(6)
        instance_image[idxs7] = _color_map(7)
        ret = {
            'binary_image': binary_image,
            'instance_image': instance_image,
        }
        return ret

