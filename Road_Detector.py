import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_road(image, showImage):
    """
    This function extracts the road in a provided image.

    @:param image : a cv2 image with a road
    @:param showImage : a boolean value to show the box surrounding the road
    @:return the box around the road
    """

    # Define area of interest in front of "car"
    row, col = image.shape[:2]
    bottomLeft = [col * 0.1, row * 0.95]
    topLeft = [col * 0.3, row * 0.4]
    bottomRight = [col * 0.9, row * 0.95]
    topRight = [col * 0.7, row * 0.4]

    corners = np.array([[bottomLeft, topLeft, topRight, bottomRight]], dtype=np.int32)

    # Create a mask for the ROI
    ROIMask = np.zeros_like(image)
    cv2.fillPoly(ROIMask, corners, (255,) * ROIMask.shape[2])
    ROI = cv2.bitwise_and(image, ROIMask)

    # Blur the image for pavement detection (this blurs out the sidewalk)
    extremeBlur = cv2.GaussianBlur(ROI, (115, 115), 0)

    # Convert to HSV
    hsvPav = cv2.cvtColor(extremeBlur, cv2.COLOR_BGR2HLS)

    # Look for pavement colors
    lowerPav = np.array([85, 95, 0])
    upperPav = np.array([130, 175, 50])
    pavementMask = cv2.inRange(hsvPav, lowerPav, upperPav)
    output = cv2.bitwise_and(ROI, ROI, mask=pavementMask)

    # Threshold of white in HSV space (look in un-blurred image)
    lowerWhite = np.array([0, 200, 0])
    upperWhite = np.array([255, 255, 255])
    whiteMask = cv2.inRange(output, lowerWhite, upperWhite)

    # Use Canny to detect solid white crosswalk lines in sidewalk mask
    canny = cv2.Canny(whiteMask, 50, 150)

    # Find longest line in canny image
    lines = cv2.HoughLinesP(canny, rho=1, theta=np.pi / 180, threshold=100, minLineLength=10, maxLineGap=80)
    numLines, _, _ = lines.shape
    longestLine = initX = finalX = 0
    for i in range(numLines):
        distance = np.sqrt((lines[i][0][0] - lines[i][0][2])**2 + (lines[i][0][1] - lines[i][0][3])**2)
        if distance > longestLine:
            initX = lines[i][0][0]
            finalX = lines[i][0][2]
            longestLine = distance

    # Find highest point in pavement mask
    colorPixels = cv2.findNonZero(output[:, :, 0])[0]

    # Create rectangle based on detected line and the highest pavement point in mask
    finalRectangle = [initX, colorPixels[0][1], finalX, image.shape[0]]

    if showImage:
        draw_rectangles(image, finalRectangle)
    return finalRectangle


def draw_rectangles(image, rectangle):
    """
    This function draws boxes on an image and displays the image.

    @:param image : a cv2 image with pedestrians
    @:param rectangles : a list of rectangles to draw on the image.
    @:return nothing but displays the image
    """
    imageCopy = image.copy()
    cv2.rectangle(imageCopy, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (0, 0, 255), 3)

    plt.imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))
    plt.show()

