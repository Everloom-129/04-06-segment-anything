DEPTH_FLAG = True
def detect_road(image_path:str,output_path:str):
    """
        This function analyzes a road scene from an image and detects various
    entities such as roads, sidewalks, and people. 
    - detection: grounding DINO model
    - segmentation:     SAM model 
    - depth prediction: DPT model 
    Detected entities are represented as LocationInfo objects and stored in a Counter.
    The function also annotates the original image with detection boxes and labels.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image with annotations and masks.

    Returns:
        obj_dict (dict): A dictionary mapping from unique entity identifiers 
        to corresponding LocationInfo objects.

    Raises:
        Exception: If the image at the given path cannot be read or processed.

    Note:
        global variables: BOX_TRESHOLD, TEXT_TRESHOLD, PED_TRESHOLD, and DEBUG,
        need to be set prior to calling this function.
    """

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image at path {image_path} could not be loaded. Skipping.")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"Failed to process image at {image_path}. Error: {e}")
        return None
    
    ROAD_SIDEWALK = ['road', 'sidewalk'] 
    P_CLASS     = ['person'] #,'bike']
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
        text_threshold=PED_TRESHOLD - 0.3
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

    if DEPTH_FLAG:
        depth_map = predict_depth(image_path,output_path)
    
    # Analyze where each person is standing
    p_surface_overlaps = []
    
    for name, person in obj_dict.items():
        if person.object_type != "person":
            continue # We only want to analyze persons
        if DEPTH_FLAG:
            person.distance = get_distance_category(depth_map,person.mask)
            person.angle   = estimate_angle(image,person)
        
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

    # (i, j, k, d) = display_mask(SAM_masks,P_masks,P_boxes,DINO_boxes,person_annotation,output_path)
    

    write_to_txt(image_path, output_path, p_surface_overlaps, obj_dict)

    plt.close()
    
    # return DINO_boxes,labels,P_labels,SAM_masks,P_masks
    return obj_dict

obj_dict= detect_road("input/scene_2.png",output_path="DINOmasked/scene_2.png")
# DINO_boxes,labels,P_labels,SAM_masks,P_masks = detect_road("input/video_0031/image_0005.png",output_path="DINOmasked/man.png")
# DINO_boxes,labels,P_labels,SAM_masks,P_masks = detect_road("JAAD_seg_by_sec/video_0268/image_0001.png",output_path="DINOmasked/video_0268/image_0001.png")
# DINO_boxes,labels,P_labels,SAM_masks,P_masks = detect_road("JAAD_seg_by_sec/video_0268/image_0003.png",output_path="DINOmasked/video_0268/image_0003.png")
# obj_dict,labels,p_labels =  detect_road("JAAD_seg_by_sec/video_0060/image_0005.png",output_path="SSS.png" )# "DINOmasked/video_0060/image_0005.png")
# obj_dict =  detect_road("input/S0710/image_0005.png",output_path="SSS.png" )# "DINOmasked/video_0060/image_0005.png")


