#!/usr/bin/env python

# Copyright (c) 2024Ã¯Â¼ÂWuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
from hobot_dnn import pyeasy_dnn as dnn
from time import time
import argparse
import logging

# Ã¦ÂÂ¥Ã¥Â¿ÂÃ¦Â¨Â¡Ã¥ÂÂÃ©ÂÂÃ§Â½Â®
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")


class BaseModel:
    def __init__(self, model_file: str) -> None:
        # å è½½BPUçbinæ¨¡å, æå°ç¸å³åæ°
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_file)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms" % (
                    1000 * (time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("â Failed to load model file: %s" % (model_file))
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(
                f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(
                f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.model_input_height, self.model_input_width = self.quantize_model[0].inputs[0].properties.shape[2:4]

    # def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
    #     """Convert BGR image to NV12 format"""
    #     begin_time = time()
    #     bgr_img = cv2.resize(bgr_img, (self.model_input_width, self.model_input_height),
    #                          interpolation=cv2.INTER_NEAREST)
    #     height, width = bgr_img.shape[0], bgr_img.shape[1]
    #     area = height * width
    #     yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    #     y = yuv420p[:area]
    #     uv_planar = yuv420p[area:].reshape((2, area // 4))
    #     uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
    #     nv12 = np.zeros_like(yuv420p)
    #     nv12[:height * width] = y
    #     nv12[height * width:] = uv_packed
    #
    #     logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
    #     return nv12
    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """Convert BGR image to NV12 format with center crop and resize"""
        begin_time = time()

        # ä¿å­åå§å¾åå°ºå¯¸
        self.original_height, self.original_width = bgr_img.shape[:2]

        # ç¡®ä¿è¾å¥æ¯1280x720åè¾¨ç
        if self.original_width != 1280 or self.original_height != 720:
            raise ValueError(
                f"è¾å¥åè¾¨çå¿é¡»æ¯1280x720ï¼å½åæ¯{self.original_width}x{self.original_height}")

        # 1. ä»1280x720ä¸­è£åªåºä¸­å¤®720x720åºå
        start_x = (1280 - 720) // 2  # (1280-720)/2=280
        cropped_720 = bgr_img[:, start_x:start_x + 720]  # é«åº¦ä¸åï¼å®½åº¦ä»1280->720

        # 2. å°720x720åç¼©å°640x640
        resized_640 = cv2.resize(cropped_720, (640, 640), interpolation=cv2.INTER_LINEAR)

        # è®¡ç®ç¼©æ¾æ¯ä¾ (720->640)
        self.scale = 640 / 720
        # è®°å½è£åªåç§»é
        self.crop_x = start_x
        self.crop_y = 0  # åç´æ¹åæ²¡æè£åª

        # è½¬æ¢ä¸ºNV12æ ¼å¼
        height, width = resized_640.shape[:2]
        area = height * width
        yuv420p = cv2.cvtColor(resized_640, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed

        logger.debug(f"Image processed: 1280x720 -> crop 720x720 -> resize 640x640")
        return nv12

    def forward(self, input_tensor: np.array) -> list:
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs

    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000 * (time() - begin_time):.2f} ms" + "\033[0m")
        return outputs


class YOLOv5_Detect(BaseModel):
    def __init__(self,
                 model_file: str,
                 conf: float,
                 iou: float,
                 nc: int,
                 anchors: list,
                 strides: list
                 ):
        super().__init__(model_file)
        # éç½®é¡¹ç®
        self.conf = conf
        self.iou = iou
        self.nc = 4
        self.strides = np.array(strides)
        input_h, input_w = self.model_input_height, self.model_input_width

        # åå§ååæ°
        self.x_scale = 1.0
        self.y_scale = 1.0
        self.original_width = 1280  # åºå®è¾å¥åè¾¨ç
        self.original_height = 720

        # stridesçgridç½æ ¼, åªéè¦çæä¸æ¬¡
        s_grid = np.stack(
            [np.tile(np.linspace(0.5, input_w // strides[0] - 0.5, input_w // strides[0]), reps=input_h // strides[0]),
             np.repeat(np.arange(0.5, input_h // strides[0] + 0.5, 1), input_w // strides[0])], axis=0).transpose(1, 0)
        self.s_grid = np.hstack([s_grid, s_grid, s_grid]).reshape(-1, 2)

        m_grid = np.stack(
            [np.tile(np.linspace(0.5, input_w // strides[1] - 0.5, input_w // strides[1]), reps=input_h // strides[1]),
             np.repeat(np.arange(0.5, input_h // strides[1] + 0.5, 1), input_w // strides[1])], axis=0).transpose(1, 0)
        self.m_grid = np.hstack([m_grid, m_grid, m_grid]).reshape(-1, 2)

        l_grid = np.stack(
            [np.tile(np.linspace(0.5, input_w // strides[2] - 0.5, input_w // strides[2]), reps=input_h // strides[2]),
             np.repeat(np.arange(0.5, input_h // strides[2] + 0.5, 1), input_w // strides[2])], axis=0).transpose(1, 0)
        self.l_grid = np.hstack([l_grid, l_grid, l_grid]).reshape(-1, 2)

        logger.info(f"{self.s_grid.shape = }  {self.m_grid.shape = }  {self.l_grid.shape = }")

        # ç¨äºå¹¿æ­çanchors, åªéè¦çæä¸æ¬¡
        anchors = np.array(anchors).reshape(3, -1)
        self.s_anchors = np.tile(anchors[0], input_w // strides[0] * input_h // strides[0]).reshape(-1, 2)
        self.m_anchors = np.tile(anchors[1], input_w // strides[1] * input_h // strides[1]).reshape(-1, 2)
        self.l_anchors = np.tile(anchors[2], input_w // strides[2] * input_h // strides[2]).reshape(-1, 2)

        logger.info(f"{self.s_anchors.shape = }  {self.m_anchors.shape = }  {self.l_anchors.shape = }")

    def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
        begin_time = time()
        # reshape
        s_pred = outputs[0].reshape([-1, (5 + self.nc)])
        m_pred = outputs[1].reshape([-1, (5 + self.nc)])
        l_pred = outputs[2].reshape([-1, (5 + self.nc)])

        # classify: Ã¥ÂÂ©Ã§ÂÂ¨numpyÃ¥ÂÂÃ©ÂÂÃ¥ÂÂÃ¦ÂÂÃ¤Â½ÂÃ¥Â®ÂÃ¦ÂÂÃ©ÂÂÃ¥ÂÂ¼Ã§Â­ÂÃ©ÂÂ (Ã¤Â¼ÂÃ¥ÂÂÃ§ÂÂ 2.0)
        s_raw_max_scores = np.max(s_pred[:, 5:], axis=1)
        s_max_scores = 1 / ((1 + np.exp(-s_pred[:, 4])) * (1 + np.exp(-s_raw_max_scores)))
        # s_max_scores = sigmoid(s_pred[:, 4])*sigmoid(s_pred[:, 4])
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf)
        s_ids = np.argmax(s_pred[s_valid_indices, 5:], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_raw_max_scores = np.max(m_pred[:, 5:], axis=1)
        m_max_scores = 1 / ((1 + np.exp(-m_pred[:, 4])) * (1 + np.exp(-m_raw_max_scores)))
        # m_max_scores = sigmoid(m_pred[:, 4])*sigmoid(m_pred[:, 4])
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf)
        m_ids = np.argmax(m_pred[m_valid_indices, 5:], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_raw_max_scores = np.max(l_pred[:, 5:], axis=1)
        l_max_scores = 1 / ((1 + np.exp(-l_pred[:, 4])) * (1 + np.exp(-l_raw_max_scores)))
        # l_max_scores = sigmoid(l_pred[:, 4])*sigmoid(l_pred[:, 4])
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf)
        l_ids = np.argmax(l_pred[l_valid_indices, 5:], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # Ã§ÂÂ¹Ã¥Â¾ÂÃ¨Â§Â£Ã§Â Â
        s_dxyhw = 1 / (1 + np.exp(-s_pred[s_valid_indices, :4]))
        # s_dxyhw = sigmoid(s_pred[s_valid_indices, :4])
        s_xy = (s_dxyhw[:, 0:2] * 2.0 + self.s_grid[s_valid_indices, :] - 1.0) * self.strides[0]
        s_wh = (s_dxyhw[:, 2:4] * 2.0) ** 2 * self.s_anchors[s_valid_indices, :]
        s_xyxy = np.concatenate([s_xy - s_wh * 0.5, s_xy + s_wh * 0.5], axis=-1)

        m_dxyhw = 1 / (1 + np.exp(-m_pred[m_valid_indices, :4]))
        # m_dxyhw = sigmoid(m_pred[m_valid_indices, :4])
        m_xy = (m_dxyhw[:, 0:2] * 2.0 + self.m_grid[m_valid_indices, :] - 1.0) * self.strides[1]
        m_wh = (m_dxyhw[:, 2:4] * 2.0) ** 2 * self.m_anchors[m_valid_indices, :]
        m_xyxy = np.concatenate([m_xy - m_wh * 0.5, m_xy + m_wh * 0.5], axis=-1)

        l_dxyhw = 1 / (1 + np.exp(-l_pred[l_valid_indices, :4]))
        # l_dxyhw = sigmoid(l_pred[l_valid_indices, :4])
        l_xy = (l_dxyhw[:, 0:2] * 2.0 + self.l_grid[l_valid_indices, :] - 1.0) * self.strides[2]
        l_wh = (l_dxyhw[:, 2:4] * 2.0) ** 2 * self.l_anchors[l_valid_indices, :]
        l_xyxy = np.concatenate([l_xy - l_wh * 0.5, l_xy + l_wh * 0.5], axis=-1)

        # Ã¥Â¤Â§Ã¤Â¸Â­Ã¥Â°ÂÃ§ÂÂ¹Ã¥Â¾ÂÃ¥Â±ÂÃ©ÂÂÃ¥ÂÂ¼Ã§Â­ÂÃ©ÂÂÃ§Â»ÂÃ¦ÂÂÃ¦ÂÂ¼Ã¦ÂÂ¥
        xyxy = np.concatenate((s_xyxy, m_xyxy, l_xyxy), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(xyxy, scores, self.conf, self.iou)

        # å¨postProcessæ¹æ³ä¸­ä¿®æ¹åæ è½¬æ¢é¨å
        bboxes = xyxy[indices].astype(np.float32)  # åè½¬æ¢ä¸ºfloatä»¥ä¾¿è¿è¡ç¼©æ¾è®¡ç®

        # 1. å°640x640åæ ç³»ä¸çåæ è½¬æ¢å720x720åæ ç³»
        bboxes /= self.scale

        # 2. å ä¸è£åªåç§»éè½¬æ¢å°1280x720åæ ç³»
        bboxes[:, 0] += self.crop_x  # x1
        bboxes[:, 2] += self.crop_x  # x2
        # yæ¹åæ²¡æè£åªï¼ä¸éè¦åç§»

        # 3. è½¬æ¢åæ´æ°åæ å¹¶ç¡®ä¿ä¸è¶åºå¾åèå´
        bboxes = bboxes.astype(np.int32)
        bboxes[:, 0:4] = np.clip(bboxes[:, 0:4], 0, [self.original_width, self.original_height,
                                                     self.original_width, self.original_height])

        return ids[indices], scores[indices], bboxes.astype(np.int32)

        # Ã¨Â¿ÂÃ¥ÂÂÃ¥ÂÂ°Ã¥ÂÂÃ¥Â§ÂÃ§ÂÂimgÃ¥Â°ÂºÃ¥ÂºÂ¦
        # bboxes = (xyxy[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])).astype(np.int32)

        # logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        # return ids[indices], scores[indices], bboxes
        # Ã¨Â°ÂÃ¦ÂÂ´Ã¥ÂÂÃ¦Â ÂÃ¦ÂÂ Ã¥Â°Â (Ã¨ÂÂÃ¨ÂÂÃ¥Â¡Â«Ã¥ÂÂÃ¥ÂÂÃ§Â¼Â©Ã¦ÂÂ¾)


"""coco_names = ['right turn','left turn','puddle','street vendor','obstacle','bad road','garbage bin','chair','pothole','car',
        'motorcycle','pedestrian','fence','gate barrier','roadblock','door','tree','plant pot','drain','stair','pole',
        'zebra cross']
"""
coco_names = ['green', 'red', 'cross', 'no_entry']

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]


def draw_detection(img: np.array, bbox, score: float, class_id: int) -> None:
    """Ã§Â»ÂÃ¥ÂÂ¶Ã¦Â£ÂÃ¦ÂµÂÃ¦Â¡ÂÃ¥ÂÂÃ¦Â ÂÃ§Â­Â¾"""
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id % 20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def detect_start(display=True):
    MODEL_PATH = '/app/pydev_demo/models/yolov5m.bin'
    VIDEO_SOURCE = '0'
    DISPLAY = display

    # ?????
    model = YOLOv5_Detect(
        model_file=MODEL_PATH,
        conf=0.6,
        iou=0.45,
        nc=4,
        anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        strides=[8, 16, 32]
    )

    # ?????????????
    try:
        video_source = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError("???????")

        # ???????????????????
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # ???????
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"????????: {actual_width}x{actual_height}")

    except Exception as e:
        logger.error(f"????????: {e}")
        exit(1)

    frame_count = 0
    start_time = time()
    return DISPLAY, frame_count, model, start_time, cap


# DISPLAY, frame_count, model, start_time,cap = detect_start(display=True)
def detect_main(DISPLAY, frame_count, model, start_time, cap):
    MP4 = True
    ret, frame = cap.read()
    if not ret:
        logger.info("Ã¨Â§ÂÃ©Â¢ÂÃ¦ÂµÂÃ§Â»ÂÃ¦ÂÂ")
        return "", False, "0"  # Ã¨Â¿ÂÃ¥ÂÂÃ©Â»ÂÃ¨Â®Â¤Ã¥ÂÂ¼

    # æ·»å å¾åç¿»è½¬ - æ°´å¹³ç¿»è½¬ï¼éåï¼
    frame = cv2.flip(frame, 0)  # åæ°1è¡¨ç¤ºæ°´å¹³ç¿»è½¬

    frame_count += 1

    # Ã¥ÂÂÃ¥Â¤ÂÃ¨Â¾ÂÃ¥ÂÂ¥Ã¦ÂÂ°Ã¦ÂÂ®
    input_tensor = model.bgr2nv12(frame)

    # Ã¦ÂÂ¨Ã§ÂÂ
    outputs = model.c2numpy(model.forward(input_tensor))

    # Ã¥ÂÂÃ¥Â¤ÂÃ§ÂÂ
    ids, scores, bboxes = model.postProcess(outputs)

    # Ã¦ÂÂÃ¥ÂÂ°Ã¦Â£ÂÃ¦ÂµÂÃ§Â»ÂÃ¦ÂÂ
    logger.info(f"Frame {frame_count} Ã¦Â£ÂÃ¦ÂµÂÃ¥ÂÂ° {len(ids)} Ã¤Â¸ÂªÃ§ÂÂ®Ã¦Â Â:")
    for class_id, score, bbox in zip(ids, scores, bboxes):
        x1, y1, x2, y2 = bbox
        logger.info(f"  {coco_names[class_id]}: Ã§Â½Â®Ã¤Â¿Â¡Ã¥ÂºÂ¦ {score:.2f}, Ã¤Â½ÂÃ§Â½Â® ({x1}, {y1}, {x2}, {y2})")

    # Ã¦Â Â¹Ã¦ÂÂ®displayÃ¥ÂÂÃ¦ÂÂ°Ã¥ÂÂ³Ã¥Â®ÂÃ¦ÂÂ¯Ã¥ÂÂ¦Ã¦ÂÂ¾Ã§Â¤ÂºÃ§Â»ÂÃ¦ÂÂÃ§ÂªÂÃ¥ÂÂ£
    if DISPLAY:
        display_frame = frame.copy()
        for class_id, score, bbox in zip(ids, scores, bboxes):
            draw_detection(display_frame, bbox, score, class_id)
        cv2.imshow('YOLOv5 Detection', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return "", False, "0"  # Ã¨Â¿ÂÃ¥ÂÂÃ§ÂÂ¹Ã¦Â®ÂÃ¥ÂÂ¼Ã¨Â¡Â¨Ã§Â¤ÂºÃ©ÂÂÃ¨Â¦ÂÃ©ÂÂÃ¥ÂÂº

    if len(ids) == 0:  # Ã¥Â¦ÂÃ¦ÂÂÃ¦Â²Â¡Ã¦ÂÂÃ¦Â£ÂÃ¦ÂµÂÃ¥ÂÂ°Ã§ÂÂ®Ã¦Â Â
        return "", MP4, "0"

    return f"{coco_names[class_id]}", MP4, f"{len(ids)}"


# f"{coco_names[class_id]}",MP4,f"{len(ids)}"=detect_main(DISPLAY, frame_count, model, start_time,cap)
# if MP4 and f"{len(ids)}" != "0":
#        result = f"{coco_names[class_id]}"
#    else:
#        result = None  # Ã¦ÂÂÃ¨ÂÂÃ¥ÂÂ¶Ã¤Â»ÂÃ©Â»ÂÃ¨Â®Â¤Ã¥ÂÂ¼
def detect_last(DISPLAY, frame_count, start_time, cap):
    # Ã¦ÂÂ§Ã¨ÂÂ½Ã§Â»ÂÃ¨Â®Â¡
    end_time = time()
    total_time = end_time - start_time
    avg_fps = frame_count / total_time
    logger.info(
        f"Ã¥Â¤ÂÃ§ÂÂÃ¦ÂÂ»Ã¥Â¸Â§Ã¦ÂÂ°: {frame_count}, Ã¦ÂÂ»Ã¦ÂÂ¶Ã©ÂÂ´: {total_time:.2f}Ã§Â§Â, Ã¥Â¹Â³Ã¥ÂÂFPS: {avg_fps:.2f}")

    # Ã©ÂÂÃ¦ÂÂ¾Ã¨ÂµÂÃ¦ÂºÂ
    cap.release()
    if DISPLAY:
        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
