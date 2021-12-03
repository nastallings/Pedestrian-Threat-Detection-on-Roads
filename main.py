import matplotlib.pyplot as plt
import Pedestrian_Detector as pd
import Feature_Extractor as fe
import Road_Detector as rd
import cv2

# load example image
X = 10
skip_rate = 1
startframe = 887
centers = []
for i in range(0,X): #check the previous X frames for direction detection
    exampleImage = cv2.imread('Data/video1/frame'+str(startframe-(skip_rate*X) + skip_rate*i)+'.jpg')
    max_width = 800

    if exampleImage.shape[1] > max_width:
        newWidth = max_width
        newHeight = exampleImage.shape[0] * (max_width/exampleImage.shape[1])
        exampleImage = cv2.resize(exampleImage, (int(newWidth), int(newHeight)))

    detected_roads = rd.detect_road(exampleImage, False)

    detected_pedestrians = pd.detect_pedestrians(exampleImage, False)

    if not len(detected_pedestrians):
        print("No Pedestrians Detected in Image.")
    else:
        for (x0, y0, x1, y1) in detected_pedestrians:
            # Get the top half of an image
            imageCopy = exampleImage.copy()
            imageCopy = imageCopy[y0:(y0 + int((y1 - y0) / 2)), x0:x1]

            fe.extract_features(imageCopy, False)

        # overlap detection

        for (x0, y0, x1, y1) in detected_roads:
            for(px0,py0, px1, py1) in detected_pedestrians:
                if (px0 >= x0 and px0 <= x1) or (px1 >= x0 and px1 <= x1):    # if X of pedestrian is within road rectangle, flag
                    print("Image Flagged for pedestrian Threat")
                    continue
                if (py0 >= y0 and py0 <= y1) or (py1 >= y0 and py1 <= y1):  # if Y of pedestrian is within road rectangle, flag
                    print("Image Flagged for pedestrian Threat")
                    continue

        centers.append(((px0 + px1)/2, (py0 + py1)/2))

#print everything
#pedestrians first
cv2.line(exampleImage, (int(centers[0][0]),int(centers[0][1])), (int(centers[-1][0]),int(centers[-1][1])), (0,0,225), 2)

for (x0, y0, x1, y1) in detected_pedestrians:
    cv2.rectangle(exampleImage, (x0, y0), (x1, y1), (0, 0, 0), 2)

#plt.imshow(cv2.cvtColor(exampleImage, cv2.COLOR_BGR2RGB))

#roads
detected_roads = detected_roads[0]
cv2.rectangle(exampleImage, (detected_roads[0], detected_roads[1]), (detected_roads[2], detected_roads[3]), (0, 0, 255), 3)

#plot pedestrian average
for (x,y) in centers:

    cv2.circle(exampleImage, (int(x),int(y)), radius=3, color=(0, 255, 0), thickness=-1)

plt.imshow(cv2.cvtColor(exampleImage, cv2.COLOR_BGR2RGB))
plt.show()
print(centers)