import matplotlib.pyplot as plt
import Pedestrian_Detector as pd
import Feature_Extractor as fe
import cv2

# load example image
exampleImage = cv2.imread('Data/video1/frame1020.jpg')

max_width = 800
if exampleImage.shape[1] > max_width:
    newWidth = max_width
    newHeight = exampleImage.shape[0] * (max_width/exampleImage.shape[1])
    exampleImage = cv2.resize(exampleImage, (int(newWidth), int(newHeight)))
detected_pedestrians = pd.detect_pedestrians(exampleImage, True)

if not len(detected_pedestrians):
    print("No Pedestrians Detected in Image.")
else:
    for (x0, y0, x1, y1) in detected_pedestrians:
        # Get the top half of an image
        exampleImage = exampleImage[y0:(y0 + int((y1 - y0) / 2)), x0:x1]

        fe.extract_features(exampleImage, True)
