import os
import cv2
import torch
import torchvision

import warnings
import numpy as np
from PIL import Image
from glob import glob

import termcolor
import matplotlib.pyplot as plt

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths to GroundingDINO and SAM checkpoints
GROUNDING_DINO_CONFIG_PATH = "/root/autodl-tmp/DINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/root/autodl-tmp/DINO/weights/groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/root/autodl-tmp/sam_vit_h_4b8939.pth"

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# Initialize GroundingDINO model
grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
    device=DEVICE
)

# Initialize SAM model and predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Classes of interest (add as needed)
CLASSES = ['person', 'sidewalk']

output_dir = 'DINOmasked'
os.makedirs(output_dir, exist_ok=True)


# Prompting SAM with ROI
def segment_ROI(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def detect_road(image_path):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )
    # NMS post process
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()
    #TODO new another structure to hold this part
    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # convert detections to masks
    detections.mask = segment_ROI(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # filter class_id==0 results
    result_masks = detections.mask[(detections.class_id==0),:,:]
    result_masks = result_masks.astype(np.uint8)

    # findout the min mask
    # find out the max mask
    max_area, max_mask = 0, np.zeros_like(result_masks[0])
    for mask in result_masks:
        area = np.sum(mask)
        if area > max_area:
            max_area = area
            max_mask = mask
    # print(max_mask)
    display_mask(result_masks, image_path,image)

    # display_mask(max_mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # write a helper function to display current mask output

if __name__ == "__main__":
    image_dir = "input"
    
    print("Start =====")
    i = 0

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(image_dir, filename)
            output_path = os.path.join(output_dir, filename)

            if not os.path.exists(output_path):
                print("Processing: ", i)
                i += 1
                result = detect_road(image_path)
                print(f"Image path: {termcolor.colored(os.path.basename(image_path), 'green')}")
                
                print(f"Detected: {termcolor.colored(result, 'blue')}")