# ---------------------------------------------------------- Libraries Import ----------------------------------------------------------

import numpy as np
import cv2
import argparse
import os
import dlib
from scipy.spatial import distance as dist
from collections import OrderedDict
from pytool import PeopleCountConf as config

# ---------------------------------------------------------- Parse Arguments ----------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
	help="input video file")
ap.add_argument("-o", "--output", type=str, default="",
	help="output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# ---------------------------------------------------------- Set Confrigrations ----------------------------------------------------------

class_file = os.path.sep.join([config.MODEL_PATH, "coco.names"])
weights_file = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
cfg_file = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# ---------------------------------------------------------- Default Status of Detecting and Tracking ----------------------------------------------------------
status = "off"
writer = None

vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

prop = cv2.CAP_PROP_FRAME_COUNT
totalFrames = int(vs.get(prop))
print(f"--> Total frames in input video {totalFrames} ")

H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))

limitIn = int(H/2 - H/9)
limitOut = int(H/2 - H/9)

print(W, H)

classNames = open(class_file).read().strip().split("\n")

# ---------------------------------------------------------- Load the serialized caffe model ----------------------------------------------------------
print("--> Loading YOLO ...")

net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

print("--> Loading YOLO ...\n --> YOLO loaded!")

# ---------------------------------------------------------- Output Layers, Centroid and Boundry Box ----------------------------------------------------------

def getOutputLayers(net):
    layerNames = net.getLayerNames()
    layerNames = [layerNames[i-1]for i in net.getUnconnectedOutLayers()]
    return layerNames

def drawBoundingBox(frame, box, centroid, color):
    (startX, startY, endX, endY) = box

    cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color, thickness=2)

def computeCentroid(box):
    (startX, startY, endX, endY) = box
    return np.array([startX + ((endX - startX)/2), startY + ((endY - startY)/2)])

# ---------------------------------------------------------- Tracking ----------------------------------------------------------

class TrackableObject:
    def __init__(self, objectID, centroid, zone):
        self.objectID = objectID
        self.centroids = [centroid]
        self.zone = zone

        self.counted = False

class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=30):

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

 
        self.maxDisappeared = maxDisappeared

        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()
 
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects

# ---------------------------------------------------------- Detection ----------------------------------------------------------
def detect(frame, layerOutputs):
    for output in layerOutputs:
        for detection in output:
          
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > config.MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, config.MIN_CONF, config.NMS_THRESH)

    for i in indices:
        if classIds[i] == 0:           #    Person Class
            
            # coordinates bbox
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = left + width
            bottom = top + height

            box = [left, top, right, bottom]
     
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(int(left), int(top), int(right), int(bottom))

            tracker.start_track(frame, rect)

            trackers.append(tracker)

            rects.append(box)

            centroid = computeCentroid(box)
           
            drawBoundingBox(frame, box, centroid, color=(0, 0, 255))

            print("Detecting...")

# ---------------------------------------------------------- centroid tracker ----------------------------------------------------------
def track(frame, trackers):
    frame_num = 0
    for tracker in trackers:
        status = "Tracking ..."
        tracker.update(frame)

        pos = tracker.get_position()

        left = int(pos.left())
        top = int(pos.top())
        right = int(pos.right())
        bottom = int(pos.bottom())

        box = [left, top, right, bottom]

        rects.append(box)

        centroid = computeCentroid(box)

        drawBoundingBox(frame, box, centroid, color=(0, 128, 255))

        if frame_num % 2==0:
              status = "Tracking ..."

        print("Tracking...")
        frame_num += 1
        
# ---------------------------------------------------------- Counting ----------------------------------------------------------
def counting(objects):
    
    global totalIn
    global totalOut

    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)
        
        if to is None:
            if centroid[1] >= H/2:
                zone = "in"
            else :
                zone = "out"
                
            to = TrackableObject(objectID, centroid, zone)

        else:
            if to.zone == "in" :
                if centroid[1] < limitOut + config.OFFSET:
                    totalOut += 1
                    to.zone = "out"
                    print("People Leaving the shop : ", totalOut)
                    
            elif to.zone == "out" :
                if centroid[1] >= limitIn + config.OFFSET:
                    totalIn += 1
                    to.zone = "in"
                    print("People Coming into the shop  : ", totalIn)
            
            to.centroids.append(centroid)

        trackableObjects[objectID] = to
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
        cv2.putText(frame, "ID : " + str(objectID), (centroid[0], centroid[1]+20), config.FONT,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA)

ct = CentroidTracker(maxDisappeared=30, maxDistance=130)

trackers = []
trackableObjects = {}

totalOut = 0
totalIn = 0

# ---------------------------------------------------------- Display ----------------------------------------------------------

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    rects = []

    if config.elapsedFrames % config.skipFrames == 0:
        classIds = []
        confidences = []
        boxes = []

        trackers = []
        status = "Detecting..."

        blob = cv2.dnn.blobFromImage(
            frame, config.SCALE_FACTOR, (config.INP_WIDTH, config.INP_HEIGHT), swapRB=True, crop=False)

        net.setInput(blob)

        layerOutputs = net.forward(getOutputLayers(net))

        detect(frame, layerOutputs)

    else:
        track(frame, trackers)

    objects = ct.update(rects)
    counting(objects)

    cv2.line(frame, (0, limitIn), (W, limitIn), (0, 255, 0), 1)
    cv2.line(frame, (0, limitOut), (W, limitOut), (255, 0, 0), 1)

    info = [
        ("Total ", int(totalIn) - int(totalOut))
    ]
    
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, ((i * 20) + 20)),
                    config.FONT, 0.7, (0,255,0), 1, cv2.LINE_AA)
    config.elapsedFrames += 1
    
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("e"):
            break

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
            (frame.shape[1], frame.shape[0]), True)

    if writer is not None:
        writer.write(frame)

print('\n ------------------------------------------ Final Result ------------------------------------------\n')
print(" --> Number of People Leaving the shop : ", totalOut)
print(" --> Number of People Coming into the shop  : ", totalIn)
print(" --> Total People inside the shop : ", int(totalIn) - int(totalOut))
	