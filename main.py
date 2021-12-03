import matplotlib.pyplot as plt
import Pedestrian_Detector as pd
import Feature_Extractor as fe
import Road_Detector as rd
import cv2
import os

class RingBuffer:
    """ class that implements a not-yet-full buffer """
    def __init__(self, size_max):
        self.max = size_max
        self.data = []

    class __Full:
        """ class that implements a full buffer """

        def append(self, x):
            """ Append an element overwriting the oldest one. """
            self.data[self.cur] = x
            self.cur = (self.cur + 1) % self.max

        def get(self):
            """ return list of elements in correct order """
            return self.data[self.cur:] + self.data[:self.cur]

    def append(self, x):
        """append an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.max:
            self.cur = 0
            # Permanently change self's class from non-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data

# load video capture from file
video = cv2.VideoCapture("/home/tyler/Desktop/SensorFusion/DetectionCode/Pedestrian-Threat-Detection-on-Roads/IMG_1564.mp4")
# window name and size


previousStates = RingBuffer(5)
centers = RingBuffer(10)
flag = "Clear"
cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
while video.isOpened():
    flag = "Clear"
    # Read video capture
    ret, frame = video.read()

    # X = 10
    # skip_rate = 1
    # startframe = 887
    # centers = []
    # for i in range(0,X): #check the previous X frames for direction detection
    #     exampleImage = cv2.imread('Data/video1/frame'+str(startframe-(skip_rate*X) + skip_rate*i)+'.jpg')
    max_width = 800

    if frame.shape[1] > max_width:
        newWidth = max_width
        newHeight = frame.shape[0] * (max_width/frame.shape[1])
        frame = cv2.resize(frame, (int(newWidth), int(newHeight)))



    detected_pedestrians = pd.detect_pedestrians(frame, False)
    if not len(detected_pedestrians):
        print("No Pedestrians Detected in Image.")
    else:
        detected_roads = rd.detect_road(frame, False)

        # for (x0, y0, x1, y1) in detected_pedestrians:
        #     # Get the top half of an image
        #     imageCopy = frame.copy()
        #     imageCopy = imageCopy[y0:(y0 + int((y1 - y0) / 2)), x0:x1]
        #
        #     fe.extract_features(imageCopy, False)

        # overlap detection
        for (x0, y0, x1, y1) in detected_roads:
            for(px0,py0, px1, py1) in detected_pedestrians:
                if ((px0 >= x0 and px0 <= x1) or (px1 >= x0 and px1 <= x1)) and ((py0 >= y0 and py0 <= y1) or (py1 >= y0 and py1 <= y1)):    # if X of pedestrian is within road rectangle, flag
                    flag = "Danger"
                    continue


        centers.append(((px0 + px1)/2, (py0 + py1)/2))

        #print everything
        #pedestrians first
        cv2.line(frame, (int(centers.get()[0][0]),int(centers.get()[0][1])), (int(centers.get()[-1][0]),int(centers.get()[-1][1])), (0,0,225), 2)

        for (x0, y0, x1, y1) in detected_pedestrians:
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), 2)

        #plt.imshow(cv2.cvtColor(exampleImage, cv2.COLOR_BGR2RGB))

        #roads
        detected_roads = detected_roads[0]
        cv2.rectangle(frame, (detected_roads[0], detected_roads[1]), (detected_roads[2], detected_roads[3]), (0, 0, 255), 3)

        #plot pedestrian average
        for (x,y) in centers.get():

            cv2.circle(frame, (int(x),int(y)), radius=3, color=(0, 255, 0), thickness=-1)

    if flag != "Danger" and "Danger" in previousStates.get():
        flag = "Warning"

    previousStates.append(flag)

    overlay = frame.copy()
    if flag == "Danger":   #Danger
        cv2.rectangle(overlay, (0, 0), (999, 999), (0, 0, 255), -1)
    elif flag == "Clear":  #Clear
        cv2.rectangle(overlay, (0, 0), (999, 999), (0, 255, 0), -1)
    elif flag == "Warning":#warning
        cv2.rectangle(overlay, (0, 0), (999, 999), (255, 255, 0), -1)

    cv2.addWeighted(overlay, 0.3, frame, .9, 0, frame)



    cv2.imshow("video", frame)
    # show one frame at a time
    key = cv2.waitKey(0)
    while key not in [ord('q'), ord(' ')]:
       key = cv2.waitKey(0)
    #Quit when 'q' is pressed
    if key == ord('q'):
       break


# Release capture object

video.release()
# Exit and distroy all windows
cv2.destroyAllWindows()


