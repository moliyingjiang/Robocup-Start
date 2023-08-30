import cv2
import sys
import time
from pathlib import Path

import easyocr
import paddle
import torch
# import torch.backends.cudnn as cudnn
import random
import numpy as np
from sklearn.cluster import KMeans

from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from models.experimental import attempt_load
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cv2
import numpy as np


def correct_skew(image):
    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(image, kernel)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the rectangle enclosing the largest contour
    max_area = 0
    max_rect = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        _, _, area = rect
        if area > max_area:
            max_area = area
            max_rect = rect

    # Compute the angle of the largest rectangle
    try:
        _, _, angle = max_rect

        # Rotate the image
        (height, width) = image.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        return rotated
    finally:
        return image


def identify_direction(image):
    # Convert image to grayscale
    alpha = 1.6
    beta = 0.8
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    cv2.imwrite('11a.jpg', binary)
    height, width = binary.shape
    middle_portion = binary[height // 4: 3 * height // 4,
                     width // 4: 3 * width // 4]  # focus on the middle half of the image

    middle_portion = correct_skew(middle_portion)
    cv2.imwrite('1a.jpg', middle_portion)
    middle_height, middle_width = middle_portion.shape

    left_half = middle_portion[:, :middle_width // 2]
    right_half = middle_portion[:, middle_width // 2:]

    lower_left_quarter = left_half[3 * middle_height // 5:, :]
    lower_right_quarter = right_half[3 * middle_height // 5:, :]

    white_pixels_left = cv2.countNonZero(lower_left_quarter)
    white_pixels_right = cv2.countNonZero(lower_right_quarter)

    white_pixels_threshold = 0.15 * (
            lower_left_quarter.shape[0] * lower_left_quarter.shape[1])  # threshold is 10% of the quarter area

    if white_pixels_left < white_pixels_threshold:
        return 'left'
    elif white_pixels_right < white_pixels_threshold:
        return 'right'
    else:
        return 'none'


def identify_direction2(image):
    # Convert image to grayscale
    alpha = 1.8
    beta = 50
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    cv2.imwrite('b11a.jpg', binary)
    height, width = binary.shape
    middle_portion = binary[height // 4: 3 * height // 4,
                     width // 4: 3 * width // 4]  # focus on the middle half of the image

    middle_portion = correct_skew(middle_portion)
    cv2.imwrite('b1a.jpg', middle_portion)
    middle_height, middle_width = middle_portion.shape

    right_half = middle_portion[:, :middle_width // 2]
    left_half = middle_portion[:, middle_width // 2:]

    quarter_height_left = left_half.shape[0] // 4
    lower_left_quarter = left_half[:quarter_height_left, :]
    quarter_height_right = right_half.shape[0] // 4
    lower_right_quarter = right_half[:quarter_height_right, :]

    white_pixels_left = cv2.countNonZero(lower_left_quarter)
    white_pixels_right = cv2.countNonZero(lower_right_quarter)

    # white_pixels_threshold = 0.15 * (lower_left_quarter.shape[0] * lower_left_quarter.shape[1])  # threshold is 10% of the quarter area

    if white_pixels_left > white_pixels_right * 1.5:
        return 'front_left'
    elif white_pixels_right * 1.002 > white_pixels_left * 1.5:
        return 'front_right'
    else:
        return 'none'
def identify_direction3(image):

    return 'left_right'


def init():
    FILE = Path(__file__).absolute()
    sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path
    device = torch.device('cuda:0')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = attempt_load('bestkaigexin.pt', map_location=device)  # load FP32 model
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
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0, 1, 2, 3, 4, 5, 6, 7], agnostic=False)
    return img, pred


if __name__ == '__main__':
    device, half, model, names, colors = init()
    video_path = "shagou.mp4"
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Failed to open {0}'
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    # ocr = PaddleOCR(det_model_dir='inference/det/en/ch_PP-OCRv3_det_infer',
    #                 rec_model_dir='inference/rec/en/ch_PP-OCRv3_rec_infer',
    #                 cls_model_dir='inference/cls/ch_ppocr_mobile_v2.0_cls_infer',
    #                 use_gpu=True,
    #                 use_angle_cls=True,
    #                 lang='en')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img, pred = predict_img([frame], device, half, model)
            # result = ocr.ocr(frame, cls=True)  # use frame instead of imgs
            ocr_results = []
            yolo_results = []
            results = reader.readtext(frame)
            for box, label, confidence in results:
                if confidence >= 0.6:
                    ocr_results.append(label)
                    if '110' in ocr_results or '120' in ocr_results or '119' in ocr_results:
                        cv2.rectangle(frame, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(box[0][0]), int(box[0][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)
            for i, det in enumerate(pred):  # detections per image
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                              frame.shape).round()  # use frame.shape instead of im0.shape
                    for *xyxy, conf, cls in reversed(det):
                        # label = f'{names[int(cls)]} {conf:.2f}'
                        label = f'{names[int(cls)]}'
                        if conf >= 0.6 and names[int(cls)] == 'right&left':
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            sign_roi = frame[y1:y2, x1:x2]
                            direction = identify_direction(sign_roi)
                            # print(f'Sign direction: {direction}')  # 打印出检测到的方向信息
                            yolo_results.append(direction)
                            # ocr_results.append(label)
                            if 'left' in yolo_results or 'right' in yolo_results or 'none' in yolo_results:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 0, 255), 3)
                                cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0,
                                            (0, 0, 255), 3)
                        elif conf >= 0.6 and names[int(cls)] == 'front_left&front_right':
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            sign_roi = frame[y1:y2, x1:x2]
                            direction = identify_direction2(sign_roi)
                            # print(f'Sign direction: {direction}')  # 打印出检测到的方向信息
                            yolo_results.append(direction)
                            # ocr_results.append(label)
                            if 'front_left' in yolo_results or 'front_right' in yolo_results or 'none' in yolo_results:
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                              (0, 0, 255), 3)

                                cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0,
                                            (0, 0, 255), 3)
                        elif conf >= 0.8 and names[int(cls)] == 'left_right':
                            x1, y1, x2, y2 = [int(xy) for xy in xyxy]
                            sign_roi = frame[y1:y2, x1:x2]
                            direction = identify_direction3(sign_roi)
                            # print(f'Sign direction: {direction}')  # 打印出检测到的方向信息
                            yolo_results.append(direction)
                            # ocr_results.append(label)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                          (0, 0, 255), 3)
                            cv2.putText(frame, direction, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0,
                                        (0, 0, 255), 3)
                        elif conf >= 0.6:
                            yolo_results.append(label)
                            plot_one_box(xyxy, frame, label=label, color=colors[int(cls)],
                                         line_thickness=3)  # draw boxes on frame
                        total_results = ocr_results + yolo_results
                        known_labels_en = ['front_left', 'front_right', 'left', 'right', 'left_right', 'fire',
                                           'first_aid',
                                           'hazmat', '110', '120', '119']
                        known_labels_cn = ['直行和向左转弯', '直行和向右转弯', '向左转弯', '向右转弯', '向左向右转弯', '火警标志', '急救标志', '危险品标志', '110', '120',
                                           '119']

                        # 创建英文标签与中文标签的映射字典
                        label_mapping = dict(zip(known_labels_en, known_labels_cn))
                        labels_new = []
                        # 遍历随机列表
                        for item in total_results:
                            if item in known_labels_en:
                                # 获取对应的中文标签
                                chinese_label = label_mapping[item]
                                labels_new.append(chinese_label)
                                # 替换输出
                                # print(labels_new)


                        def red(frame):
                            frame[:, :, 2] += 5  # 红色通道加上20
                            frame[:, :, 1] -= 5  # 绿色通道减去10
                            frame[:, :, 0] -= 5  # 蓝色通道减去10
                            cv2.putText(frame, 'False', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                        def green(frame):
                            frame[:, :, 2] -= 5  # 红色通道减去10
                            frame[:, :, 1] += 5  # 绿色通道加上20
                            frame[:, :, 0] -= 5  # 蓝色通道减去10
                            cv2.putText(frame, 'True', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                        if len(total_results) != 0:
                            if '119' in labels_new and '火警标志' in labels_new:
                                green(frame)
                                print("火警标志 电话119 正确")
                            if '119' in labels_new and '危险品标志' in labels_new:
                                red(frame)
                                print("危险品标志 电话119 错误")
                            if '120' in labels_new and '火警标志' in labels_new:
                                red(frame)
                                print("火警标志 电话120 错误")
                            if '120' in labels_new and '危险品标志' in labels_new:
                                red(frame)
                                print("危险品标志 电话120 错误")
                            if '110' in labels_new and '火警标志' in labels_new:
                                red(frame)
                                print("火警标志 电话110 错误")
                            if '110' in labels_new and '危险品标志' in labels_new:
                                green(frame)
                                print("危险品标志 电话110 正确")
                            if '119' in labels_new and '急救标志' in labels_new:
                                red(frame)
                                print("急救标志 电话119 错误")
                            if '120' in labels_new and '急救标志' in labels_new:
                                green(frame)
                                print("急救标志 电话120 正确")
                            if '110' in labels_new and '急救标志' in labels_new:
                                red(frame)
                                print("急救标志 电话110 错误")


                            def remove_labels(labels_to_remove):
                                labels_ok = []
                                # all_labels = ['前左', '前右', '左', '右', '左右', '火警标志', '急救', '危险物质', '110', '120', '119']
                                for label in labels_new:
                                    if label not in labels_to_remove:
                                        labels_ok.append(label)
                                        print(labels_ok)


                            labels_to_remove = ['119', '火警标志', '危险品标志', '120', '急救标志', '110']
                            # labels_new = ['119', '火警标志', '测试', '120', '110']
                            remove_labels(labels_to_remove)
            # Display the frame with bounding boxes and labels
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
