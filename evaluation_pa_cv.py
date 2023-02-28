import math
import os
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import cv2
from pycocotools import mask as maskUtils

from Transformer import Transformer

from DB_cfg import *
from my_utils import encode_bool_masks
from mAP_example import match_masks, compute_eval2
import LabelData

# ...............................


# Model..............................

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg



#predictor = DefaultPredictor(cfg)
#outputs = predictor(P)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# print(outputs["instances"].scores)
# print(outputs["instances"].pred_masks)

# copy results from gpu to cpu
#insts = outputs["instances"].to("cpu")


# ...............................



# def decode_bool_masks(masks):
#     if type(masks) == list:
#         masks = maskUtils.decode(masks)  # (H, W, N), column major
#         masks = np.ascontiguousarray(np.transpose(masks, [2, 0, 1]))  # (N, H, W), row-major
#     return masks.astype(np.bool)  # bool


def run_eval(gt_path, detect_path):
    iou_threshold = np.arange(0.5, 1.0, 0.05)
    total_ap = 0
    for threshold in iou_threshold:
        N_GT, TP_scores, FP_scores = 0, [], []
        # for i, pkl in enumerate(gt_path):
        for i, pkl in enumerate(os.listdir(gt_path)):
            print(f'\r{i}, {pkl}', end='')
            with open(os.path.join(gt_path, pkl), 'rb') as f:
                GT = pickle.load(f)
                # print(GT)
            gt_masks = np.zeros((len(GT['ListRegion']), GT["MapPixel2RegionId"].shape[0], GT["MapPixel2RegionId"].shape[1]), dtype=bool)
            for i in range(len(GT['ListRegion'])):
                t_listRegionId = GT['ListRegion'][i].listRegionId
                for j in range(len(t_listRegionId)):
                    gt_masks[i, GT['MapPixel2RegionId'] == t_listRegionId[j]] = True
            with open(os.path.join(detect_path, os.path.basename(pkl)), 'rb') as f:
                DT = pickle.load(f)


            scores = DT['scores']
            dt_masks = DT['masks']
            flagTP, gt_used = match_masks(dt_masks, gt_masks, threshold)
            N_GT += len(gt_used)
            TP_scores.append(scores[flagTP])
            FP_scores.append(scores[np.logical_not(flagTP)])
        TP_scores = np.concatenate(TP_scores)
        FP_scores = np.concatenate(FP_scores)
        print('compute_eval')
        # calculate precision, recall and VOC AP
        ap, mrec, mpre = compute_eval2(N_GT, TP_scores, FP_scores, 'result/pretrained/PAMaskRcnnalldataeval_aug')
        print(f'AP{int(threshold*100)}={ap}')
        total_ap += ap
    print(f'AP = {total_ap / 10}')


if __name__ == '__main__':


    gt_path = 'data/PA/instance_label'
    pred_path = 'result/pretrained/results_PAMaskRcnnalldata'
    run_eval(gt_path, pred_path)

