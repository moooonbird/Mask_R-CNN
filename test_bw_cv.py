import math
import os
import pickle
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import cv2
from pycocotools import mask as maskUtils

# from Transformer import Transformer

# from DB_cfg import *
# from my_utils import encode_bool_masks

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


def load_mask_rcnn_model(model_name: str, part):
    cfg = get_cfg()

    # gpu index
    # cfg.MODEL.DEVICE = "cuda:1" if model_name != 'fisheye20220211-2' else "cuda:0"
    cfg.MODEL.DEVICE = "cuda:1"

    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # set threshold for this model
    # print(f'{model_name}/model_final_{part}.pth')
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = f'{model_name}/model_final_{part}.pth'

    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_000.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_0003239.pth'
    # cfg.MODEL.WEIGHTS = f'{model_name}/model_final_02.pth'
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo


    # change the last nms
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3

    # more topk per image
    cfg.TEST.DETECTIONS_PER_IMAGE = 10000

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # return this predictor
    return DefaultPredictor(cfg)


# predictor = DefaultPredictor(cfg)
# outputs = predictor(P)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
# print(outputs["instances"].pred_classes)
# print(outputs["instances"].pred_boxes)
# print(outputs["instances"].scores)
# print(outputs["instances"].pred_masks)

# copy results from gpu to cpu
#insts = outputs["instances"].to("cpu")


# ...............................


# def run_detection(model_name: str):
#     model = load_mask_rcnn_model(model_name)
#     print('model:', model, 'loaded......', flush=True)

#     for dbi, DB in enumerate(DB_LIST):
#         if dbi not in [7, 8, 11]:
#             print(f'{DB}: Maybe results are not good(?)', flush=True)
#         for vi in range(1, VideoNums[dbi] + 1):
#             for pim in Path('Combine_All').glob(f'image/{dbi}_{vi}_*.jpg'):
#                 print(f'\r{dbi} {vi}', end='')
#                 # Read image
#                 I = cv2.imread(str(pim))
#                 # Build Transform
#                 # Transform image
#                 T_I_list = [I]
#                 # Detect objects on transformed images with first hard NMS
#                 dets_list = [model(T_I) for T_I in T_I_list]
#                 # detectron2 format
#                 # copy results from gpu to cpu
#                 dets_list = [dets["instances"].to("cpu") for dets in dets_list]
#                 # only human
#                 dets_list = [dets[dets.pred_classes == 0] for dets in dets_list]
#                 # to np array, drop classes
#                 dets_list = [{'scores': dets.scores.numpy(), 'masks': dets.pred_masks.numpy()} for dets in dets_list]
#                 # encode masks
#                 dets_list = [{'scores': dets['scores'], 'masks': encode_bool_masks(dets['masks'])} for dets in dets_list]
#                 result_name = 'results' + (model_name[-1] if model_name[-2] == '-' else '')
#                 Path(f'result/pretrained/{result_name}/').mkdir(parents=True, exist_ok=True)
#                 with Path(f'result/pretrained/{result_name}/{pim.name[:-4]}.pkl').open('wb') as f:
#                     pickle.dump(dets_list[0], f)

def visualization(dets, name, ori):
    mask = dets['masks'].astype(int)
    bbox = dets['bbox']
    for i in range(mask.shape[0]):
        m = mask[i]
        b = bbox[i]
        rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        rgb[m == 1] = [255, 0, 0]
        cv2.rectangle(rgb, (b[0], b[1]), (b[2], b[3]), color=(0, 255, 0), thickness=5)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, np.hstack((ori, rgb)))
        cv2.waitKey()
    cv2.destroyAllWindows()


from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
import time

def run_detection(model_name:str, data_path, cross_data_path, part):
    model = load_mask_rcnn_model(model_name, part)
    print('model:', model, 'loaded......', flush=True)
    filedata = open(cross_data_path, 'rb')
    alldatalist = pickle.load(filedata)
    val_idx = alldatalist['val_idx'][part]
    total_time = 0
    # _, cross_data = pickle.load(filedata)
    # val_idx = cross_data[part]

    # val_data = [os.listdir(data_path)[ix] for ix in val_idx]
    building_metadata = MetadataCatalog.get('123')
    for ix, val in enumerate(val_idx):
        img = cv2.imread(data_path+val)
        print(val)
        start_time = time.clock()
        dets_list = [model(img)]
        total_time += (time.clock()-start_time)
        # try:
        #     mask = model(img)['pred_masks'].numpy()
        #     print('\n mask:', mask)
        # except:
        #     continue
        # print(mask)
        dets_list = [dets["instances"].to("cpu") for dets in dets_list]
        # print(dets_list)
        v = Visualizer(img[:, :, ::-1],
                        scale=1,
                        metadata=building_metadata,
                        instance_mode = ColorMode.IMAGE_BW,
        )
        v = v.draw_instance_predictions(dets_list[0])
        Path(f'result/pretrained/BWMaskRcnnpred/').mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f'result/pretrained/BWMaskRcnnpred/{val}', v.get_image()[:, :, ::-1])
        # # plt.imshow(v.get_image[:, :, ::-1])
        # cv2.imshow('', v.get_image()[:, :, ::-1])
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        dets_list = [dets[dets.pred_classes == 0] for dets in dets_list]
        # print(dets_list)

        dets_list = [{'scores': dets.scores.numpy(), 'masks': dets.pred_masks.numpy(), 'bbox':dets.pred_boxes.tensor.numpy()} for dets in dets_list]
        # print(dets_list[0]['scores'])
        # print(dets_list[0]['bbox'])
        # print(dets_list[0]['masks'].shape)
        # visualization(dets_list[0], val, img)
        result_name = 'results_' + 'BWMaskRcnn'
        Path(f'result/pretrained/{result_name}/').mkdir(parents=True, exist_ok=True)
        with Path(f'result/pretrained/{result_name}/{val[:-4]}.pkl').open('wb') as f:
            pickle.dump(dets_list[0], f)
    
    
    return total_time

def run_eval(gt_path, detect_path):
    for pkl in os.listdir(gt_path):
        with Path(os.path.join(gt_path, pkl)).open('wb') as f:
            X = pickle.load(f)


if __name__ == '__main__':
    cross_data_path = './CV_weight/BW_panoptic_cross_data_214.pkl'
    data_path = './data/CutImage/image/'
    # mode = 'Baseline'
    total_time = 0
    for i in range(0, 4):
        total_time += run_detection('checkpoints/BWMaskRcnn', data_path=data_path, cross_data_path=cross_data_path, part=i)
    print(f"Total_time = {total_time}")
    # print(f"Inference_time(s/img) = {total_time / 216}")
    print(f"Inference_time(s/img) = {total_time / 214}")

    # img = cv2.imread('./data/image/04561074_20190502_27_up.jpg')
    # # print(img)
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = './BWMaskRcnn220615/model_final_1.pth'
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(img)
    # print(outputs)
    # outputs = model(img)
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    # print(outputs["instances"].scores)
    # print(outputs["instances"].pred_masks)   
    # run_detection('')
    # pool = Pool()
    # aa = pool.map_async(run_detection, ['pretrained'])  # gpu 1
    # pool.map_async(run_detection, ['fisheye20220211-2'])  # gpu 0
    # aa.get()
    # pool.map_async(run_detection, ['fisheye20220211-3'])  # gpu 1
    # pool.close()
    # pool.join()
