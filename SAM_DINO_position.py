import os
import cv2
# filter some annoying debug info
import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
import supervision as sv

import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter

import termcolor
import matplotlib.pyplot as plt

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
#TODO name!
from groundingdino.util.inference import load_model, load_image, predict, annotate

import SAM_utility

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Paths to GroundingDINO and SAM checkpoints
GROUNDING_DINO_CONFIG_PATH = "/root/autodl-tmp/DINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/root/autodl-tmp/DINO/weights/groundingdino_swint_ogc.pth"
model_type = "default"
SAM_CHECKPOINT_PATH = "/root/autodl-tmp/sam_vit_h_4b8939.pth"

# Predict classes and hyper-param for GroundingDINO
BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# Initialize GroundingDINO model
grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, 
    device=DEVICE
)

# Initialize SAM model and predictor
sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Prompting SAM with Region of Interest
def segment_ROI(sam_predictor: SamPredictor, image: np.ndarray, boxes: np.ndarray):
    sam_predictor.set_image(image)
    result_masks = []
    for box in boxes:
        masks_np, scores_np, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=True,
        )
        # Remove the following line to get all the person masks
        # index = np.argmax(scores_np) 
        # Add all masks to the result, not just the one with the highest score
        for mask in masks_np:
            result_masks.append(mask)

    return np.array(result_masks)


def detect_road(image_path,output_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at path {image_path} could not be loaded. Skipping.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Failed to process image at {image_path}. Error: {e}")
        return None
    
    TEXT_PROMPT = "road . sidewalk"
    ROAD_SIDEWALK = ['road', 'sidewalk'] 
    P_CLASS     = ['person',]
    # the person label lower gDINO's performance
    # so I split them

    # detect road and sidewalk
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes = ROAD_SIDEWALK,
        box_threshold= BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    detections = nms_processing(detections)
    # detect person 
    p_detections = grounding_dino_model.predict_with_classes(
        image = image,
        classes = P_CLASS , 
        box_threshold= BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    p_detections = nms_processing(p_detections)

    box_annotator = sv.BoxAnnotator()
    person_annotator = sv.BoxAnnotator()

    labels = [
        f"{ROAD_SIDEWALK[class_id]} {i} {confidence:0.2f}" 
        for i, (_, _, confidence, class_id, _) in enumerate(detections)]

    P_labels = [
        f"{P_CLASS[class_id]} {i} {confidence:0.2f}" 
        for i, (_, _, confidence, class_id, _) in enumerate(p_detections)]

    DINO_boxes = np.array(detections.xyxy)
    P_boxes    = np.array(p_detections.xyxy)
    
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections ,labels=labels)
    if DEBUG:
        sv.plot_image(annotated_frame, (16, 16))
    person_annotation = person_annotator.annotate(scene=annotated_frame,detections= p_detections,labels= P_labels)
    if DEBUG:
        sv.plot_image(person_annotation, (16, 16))
    # cv2.imwrite("annotated_image.jpg", annotated_frame)
    
    SAM_masks = segment_ROI(sam_predictor,image,DINO_boxes)
    P_masks = segment_ROI(sam_predictor,image,DINO_boxes)

    # Create a list of LocationInfo objects for each detected object
    obj_dict = Counter()
    
    for i, (box, label, mask) in enumerate(zip(DINO_boxes, labels, SAM_masks)):
        object_type, id, confidence   = label.split(' ')
        index = object_type +id
        obj_dict[index] =  (LocationInfo(object_type, int(id), box, mask,confidence)) 

    for i, (box, label, mask) in enumerate(zip(P_boxes, P_labels, P_masks)):
        object_type, id, confidence = label.split(' ')
        index = object_type+id
        obj_dict[index] = (LocationInfo(object_type, int(id), box, mask,confidence)) 

    # Analyze where each person is standing
    p_surface_overlaps = []

    for name, person in obj_dict.items():
        if person.object_type != "person":
            continue # We only want to analyze persons

        overlaps = []
        for name, surface in obj_dict.items():
            # We only want to analyze surfaces (road or sidewalk)
            if surface.object_type not in ROAD_SIDEWALK: 
                continue

            # Check if the person and the surface overlap
            overlap, _ = is_overlap(person.mask, surface.mask)
            if overlap:
                overlaps.append(surface)

        p_surface_overlaps.append((person, overlaps))


    if DEBUG:
        # Print out the analysis results
        for person, surfaces in p_surface_overlaps:
            if surfaces:
                surface_str = ', '.join([f"{surface.object_type} {surface.id}" for surface in surfaces])
                print(f"Person {person.id} is on the {surface_str}")
            else:
                print(f"Person {person.id} is not on any detected surface")

    (i, j, k, d) = display_mask(SAM_masks,P_masks,P_boxes,DINO_boxes,person_annotation,output_path)
    
    output_dir = Path(output_path).parent
    img_name = image_path[-4:-1]
    txt_name = "Info_Video_"+ img_name +".txt"
    txt_path = os.path.join(output_dir, txt_name) 
    write_to_txt(txt_path, img_name, p_surface_overlaps, (i, j, k, d), labels, P_labels)

    plt.close()
    
    return DINO_boxes,labels,P_labels,SAM_masks,P_masks




def main():
    image_dir = Path("input")
    output_dir = Path('DINOmasked')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Start =====")
    i = 1
    # for filename in os.listdir(image_dir):
    #     if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
    #         image_path = os.path.join(image_dir, filename)
    #         output_path = os.path.join(output_dir, filename)
    #         if not os.path.exists(output_path):
    #             print("Processing: ", i)
    #             i += 1 # improve this in more elegant way
    #             print(f"Image path: {termcolor.colored(os.path.basename(image_path), 'green')}")
    #             result = detect_road(image_path)
    #             print(f"Detected: {termcolor.colored(result, 'blue')}")

    # Use rglob to recursively find all image files
    for image_path in image_dir.rglob('*'):
        if SAM_utility.is_image_file(str(image_path)):
            relative_path = image_path.relative_to(image_dir)

            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True,exist_ok=True)

            if not output_path.exists():
                print("Processing: ", i)
                i += 1
                print(f"Image path: {termcolor.colored(os.path.basename(str(image_path)), 'green')}")
                result = detect_road(str(image_path),str(output_path))
                if result is not None:
                    print(f"Detected: {termcolor.colored(result, 'blue')}")
                else:
                    fail_str = "failed to detect result"
                    print(f" {termcolor.colored(fail_str, 'blue')}")



if __name__ == "__main__":
    # text program to make sure the label works
    # DINO_boxes,labels = detect_road("input/scene_2.png")
    # print(DINO_boxes, labels)   
    '''
    [[  1.7368164 187.55162   893.4925    430.34235  ]
    [351.7953    195.08667   893.7616    429.8313   ]] ['road 0.50', 'sidewalk 0.31']
    '''
    main()