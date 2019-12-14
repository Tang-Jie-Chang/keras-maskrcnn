# import keras
import keras

# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import binary_mask_to_rle

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_v0.2.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')
#print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow', 
                   10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'}

coco_dt = []

for imgid in range(100):
    image = read_image_bgr("test_images/" + coco.loadImgs(ids=imgid)[0]['file_name'])
    image = preprocess_image(image)
    image, scale = resize_image(image)
    outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    boxes  = outputs[-4][0]
    scores = outputs[-3][0]
    labels = outputs[-2][0]
    masks  = outputs[-1][0]

    # correct for image scale
    boxes /= scale

    # visualize detections
    for box, score, label, mask in zip(boxes, scores, labels, masks):
        if score < 0.5:
            break
        pred = {}
        pred['image_id'] = imgid # this imgid must be same as the key of test.json
        pred['category_id'] = label
        pred['segmentation'] = binary_mask_to_rle(mask) # save binary mask to RLE, e.g. 512x512 -> rle
        pred['score'] = score
        coco_dt.append(pred)
    
    

with open("submission.json", "w") as f:
    json.dump(coco_dt, f)