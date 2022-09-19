import cv2

# base path of YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

SCALE_FACTOR = 1 / 255.0

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# Frames

elapsedFrames = 0
skipFrames = 10

# Input Parameters

INP_WIDTH = 416 
INP_HEIGHT = 416 
OFFSET = 0.6
# Font

FONT = cv2.FONT_HERSHEY_COMPLEX