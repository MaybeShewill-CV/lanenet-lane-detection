#!/usr/bin/env bash

set -eux

PYTHONPATH=$(pwd) python freeze_lanenet_model --weights_path ./ckpt_file_path --save_path ./pb_file_path

MNNConverter -f TF --modelFile ./pb_file_path --MNNModel ./lanenet_model.mnn --bizCode MNN
