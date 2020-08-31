import logging
import json
from PIL import Image
import base64
import io
import numpy as np
from numpy import random
import cv2

# for yolov5 import
import sys
sys.path.append("/yolo/yolov5")
import torch

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils import datasets

# setting
device = 'cpu'
weights = '/yolo/yolov5/maskdetect_yolov5x.pt' # self trained model


def detect_mask(data):
    image = Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
    cvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # resize for inference
    resize_img = datasets.letterbox(cvImage, new_shape=640)[0]
    resize_img = np.transpose(resize_img, (2,0,1)) # (H,W,Ch) > (Ch,H,W)
    img = torch.from_numpy(resize_img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    img = img.unsqueeze(0)
    logging.info(img.shape)

    # load model
    model = attempt_load(weights, map_location=device)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    logging.info("names: %s", names)

    # Inference
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=0, agnostic=True)

    # input image
    im0 = cvImage

    s = '%gx%g ' % img.shape[2:]  # print string
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    det = pred[0]
    logging.info(det)
    if det is not None:
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        logging.info(det.size(0))
        # Print results
        uni = det[:, -1].int()
        unit = torch.unique(uni)
        for c in uni:
            n = (det[:, -1] == c).sum()  # detections per class
            print(n)
            s += '%g %ss, ' % (n, names[int(c)])  # add to string
            print(s)

        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = '%s %.2f' % (names[int(cls)], conf)
            print(label)
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        logging.info(im0.shape)
    

    # image encoding
    #img_array = np.transpose(img, (1,2,0))
    img_encoded = cv2.imencode('.jpg', im0)[1] # (H,W,Ch)
    encodedStr = base64.b64encode(img_encoded).decode('ascii')
    mask_img = encodedStr
    return mask_img


def main(ctx, msg):
    logging.info("***************** mask detect start *****************")
  
    # recieve a image from raw-to-jpg
    rmsg = json.loads(msg)
    rmsg['mask'] = detect_mask(rmsg['data'])

    logging.info("***************** mask detect end *****************")
    rmsg = json.dumps(rmsg).encode('utf-8')
    ctx.send(rmsg)