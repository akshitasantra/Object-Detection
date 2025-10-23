import os
import torchvision
from torchvision import  transforms 
import torch
from torch import no_grad

import requests

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_predictions(pred,threshold=0.8,objects=None ):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """


    # build predicted_classes list
    predicted_classes = [
        (COCO_INSTANCE_CATEGORY_NAMES[i], p, ((box[0], box[1]), (box[2], box[3])))
        for i, p, box in zip(
            pred[0]['labels'].numpy(),
            pred[0]['scores'].detach().numpy(),
            pred[0]['boxes'].detach().numpy()
        )
        if p > threshold  # filter by threshold
    ]

    # filter by objects if provided
    if objects and predicted_classes:
        predicted_classes = [
            (name, p, box) for name, p, box in predicted_classes if name in objects
        ]

    return predicted_classes


def draw_box(predicted_classes,image,rect_th= 2,text_size= .6,text_th=1):
    """
    draws box around each object 
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface 
   
    """

    img = (np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0, 1) * 255).astype(np.uint8).copy()
    
    for predicted_class in predicted_classes:
        label = predicted_class[0]
        probability = predicted_class[1]
        box = predicted_class[2]

        # Convert coordinates to int tuples for OpenCV
        pt1 = tuple(map(int, box[0]))
        pt2 = tuple(map(int, box[1]))

        cv2.rectangle(img, pt1, pt2, (0, 255, 0), rect_th)
        cv2.putText(img, f"{label}: {probability:.2f}", pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    del img
    del image


def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)


model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

for name, param in model_.named_parameters():
    param.requires_grad = False
print("done")

def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
len(COCO_INSTANCE_CATEGORY_NAMES)

cwd = os.getcwd()
half = 0.5
img_path='watts_photos2758112663727581126637_b5d4d192d4_b.jpeg'
image = Image.open(os.path.join(cwd, img_path))
image.resize([int(half * s) for s in image.size])
plt.imshow(np.array(image))
image = Image.open(img_path)

image.resize( [int(half * s) for s in image.size] )

transform = transforms.Compose([transforms.ToTensor()])

img = transform(image)

pred = model([img])

pred[0]['labels']
pred[0]['scores']

index=pred[0]['labels'][0].item()
COCO_INSTANCE_CATEGORY_NAMES[index]

bounding_box=pred[0]['boxes'][0].tolist()

t,l,r,b=[round(x) for x in bounding_box]

img_plot=(np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8)
cv2.rectangle(img_plot,(t,l),(r,b),(0, 255, 0), 10) # Draw Rectangle with the coordinates
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
pred_thresh=get_predictions(pred,threshold=.995)
draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
del pred_thresh
plt.show()


