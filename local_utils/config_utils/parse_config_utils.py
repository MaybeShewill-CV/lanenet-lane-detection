#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/13 上午11:17
# @Author  : PaddlePaddle
# @Site    : https://github.com/PaddlePaddle/PaddleSeg
# @File    : parse_config_utils.py
# @IDE: PyCharm
"""
Parse config utils
"""
import os
import yaml
import json
import codecs
from ast import literal_eval


class Config(dict):
    """
    Config class
    """
    def __init__(self, *args, **kwargs):
        """
        init class
        :param args:
        :param kwargs:
        """
        if 'config_path' in kwargs:
            config_content = self._load_config_file(kwargs['config_path'])
            super(Config, self).__init__(config_content)
        else:
            super(Config, self).__init__(*args, **kwargs)
        self.immutable = False

    def __setattr__(self, key, value, create_if_not_exist=True):
        """

        :param key:
        :param value:
        :param create_if_not_exist:
        :return:
        """
        if key in ["immutable"]:
            self.__dict__[key] = value
            return

        t = self
        keylist = key.split(".")
        for k in keylist[:-1]:
            t = t.__getattr__(k, create_if_not_exist)

        t.__getattr__(keylist[-1], create_if_not_exist)
        t[keylist[-1]] = value

    def __getattr__(self, key, create_if_not_exist=True):
        """

        :param key:
        :param create_if_not_exist:
        :return:
        """
        if key in ["immutable"]:
            return self.__dict__[key]

        if key not in self:
            if not create_if_not_exist:
                raise KeyError
            self[key] = Config()
        if isinstance(self[key], dict):
            self[key] = Config(self[key])
        return self[key]

    def __setitem__(self, key, value):
        """

        :param key:
        :param value:
        :return:
        """
        if self.immutable:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but SegConfig is immutable'.
                format(key, value))
        #
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except ValueError:
                pass
            except SyntaxError:
                pass
        super(Config, self).__setitem__(key, value)

    @staticmethod
    def _load_config_file(config_file_path):
        """

        :param config_file_path
        :return:
        """
        if not os.access(config_file_path, os.R_OK):
            raise OSError('Config file: {:s}, can not be read'.format(config_file_path))
        with open(config_file_path, 'r') as f:
            config_content = yaml.safe_load(f)

        return config_content

    def update_from_config(self, other):
        """

        :param other:
        :return:
        """
        if isinstance(other, dict):
            other = Config(other)
        assert isinstance(other, Config)
        diclist = [("", other)]
        while len(diclist):
            prefix, tdic = diclist[0]
            diclist = diclist[1:]
            for key, value in tdic.items():
                key = "{}.{}".format(prefix, key) if prefix else key
                if isinstance(value, dict):
                    diclist.append((key, value))
                    continue
                try:
                    self.__setattr__(key, value, create_if_not_exist=False)
                except KeyError:
                    raise KeyError('Non-existent config key: {}'.format(key))

    def check_and_infer(self):
        """

        :return:
        """
        if self.DATASET.IMAGE_TYPE in ['rgb', 'gray']:
            self.DATASET.DATA_DIM = 3
        elif self.DATASET.IMAGE_TYPE in ['rgba']:
            self.DATASET.DATA_DIM = 4
        else:
            raise KeyError(
                'DATASET.IMAGE_TYPE config error, only support `rgb`, `gray` and `rgba`'
            )
        if self.MEAN is not None:
            self.DATASET.PADDING_VALUE = [x * 255.0 for x in self.MEAN]

        if not self.TRAIN_CROP_SIZE:
            raise ValueError(
                'TRAIN_CROP_SIZE is empty! Please set a pair of values in format (width, height)'
            )

        if not self.EVAL_CROP_SIZE:
            raise ValueError(
                'EVAL_CROP_SIZE is empty! Please set a pair of values in format (width, height)'
            )

        # Ensure file list is use UTF-8 encoding
        train_sets = codecs.open(self.DATASET.TRAIN_FILE_LIST, 'r', 'utf-8').readlines()
        val_sets = codecs.open(self.DATASET.VAL_FILE_LIST, 'r', 'utf-8').readlines()
        test_sets = codecs.open(self.DATASET.TEST_FILE_LIST, 'r', 'utf-8').readlines()
        self.DATASET.TRAIN_TOTAL_IMAGES = len(train_sets)
        self.DATASET.VAL_TOTAL_IMAGES = len(val_sets)
        self.DATASET.TEST_TOTAL_IMAGES = len(test_sets)

        if self.MODEL.MODEL_NAME == 'icnet' and \
                len(self.MODEL.MULTI_LOSS_WEIGHT) != 3:
            self.MODEL.MULTI_LOSS_WEIGHT = [1.0, 0.4, 0.16]

    def update_from_list(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command line options config format error! Please check it: {}".
                format(config_list))
        for key, value in zip(config_list[0::2], config_list[1::2]):
            try:
                self.__setattr__(key, value, create_if_not_exist=False)
            except KeyError:
                raise KeyError('Non-existent config key: {}'.format(key))

    def update_from_file(self, config_file):
        """

        :param config_file:
        :return:
        """
        with codecs.open(config_file, 'r', 'utf-8') as f:
            dic = yaml.safe_load(f)
        self.update_from_config(dic)

    def set_immutable(self, immutable):
        """

        :param immutable:
        :return:
        """
        self.immutable = immutable
        for value in self.values():
            if isinstance(value, Config):
                value.set_immutable(immutable)

    def is_immutable(self):
        """

        :return:
        """
        return self.immutable

    def dump_to_json_file(self, f_obj):
        """

        :param f_obj:
        :return:
        """
        origin_dict = dict()
        for key, val in self.items():
            if isinstance(val, Config):
                origin_dict.update({key: dict(val)})
            elif isinstance(val, dict):
                origin_dict.update({key: val})
            else:
                raise TypeError('Not supported type {}'.format(type(val)))
        return json.dump(origin_dict, f_obj)


lanenet_cfg = Config(config_path='./config/tusimple_lanenet.yaml')


if __name__ == '__main__':
    """
    test
    """
    train_epoch_nums = cityscapes_cfg_v2.TRAIN.EPOCH_NUMS
    batch_size = cityscapes_cfg_v2.TRAIN.BATCH_SIZE
    model_save_dir = cityscapes_cfg_v2.TRAIN.MODEL_SAVE_DIR
    snapshot_epoch = cityscapes_cfg_v2.TRAIN.SNAPSHOT_EPOCH
    enable_miou = cityscapes_cfg_v2.TRAIN.COMPUTE_MIOU.ENABLE
    with open('./test_config_file.json', 'w', encoding='utf-8') as file:
        cityscapes_cfg_v2.dump_to_json_file(file)
