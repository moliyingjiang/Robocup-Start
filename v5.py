import cv2
import sys
import time
from pathlib import Path

import torch
import random
import numpy as np

from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from models.experimental import attempt_load
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import yaml
import cv2
import numpy as np

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def init():
    FILE = Path(__file__).absolute()
    sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
    device = torch.device('cuda:0')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load(config['weight'], map_location=device)  # load FP32 model
    imgsz = check_img_size(640, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
    # cudnn.benchmark = True  # set True to speed up constant image size inference
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    img01 = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img01.half() if half else img01) if device.type != 'cpu' else None  # run once
    return device, half, model, names, colors


def predict_img(imgs, device, half, model):
    img = [letterbox(x, new_shape=640, auto=True)[0] for x in imgs]
    # Stack
    img = np.stack(img, 0)
    # Convert
    img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=False)[0]
    # Apply NMS
    # 在此设置自己所要用的标签序号(根据训练时候的配置)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=config['classes'], agnostic=False)
    return img, pred
