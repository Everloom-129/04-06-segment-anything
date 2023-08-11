import cv2

# def get_cv2_image(image_path: str):
#     """
#     This function loads an image from a given path and converts it from BGR to RGB.

#     Args:
#     - image_path: The path to the image.

#     Returns:
#     - The converted image if successful, or None if the image could not be loaded or processed.
#     """
#     try:
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Image at path {image_path} could not be loaded. Skipping.")
#             return None
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         return image

#     except Exception as e:
#         print(f"Failed to process image at {image_path}. Error: {e}")
#         return None

def is_image_file(filename):
    IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(filename.endswith(extension) for extension in IMAGE_EXT)
