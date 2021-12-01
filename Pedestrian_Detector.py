from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_pedestrians(image, drawRectangles):
    """
    This function detects pedestrians in a provided image.

    @:param image : a cv2 image to look for pedestrians
    @:param drawRectangles : a boolean value to draw rectangles on the image
    @:return a list of rectangles identifying the locations of pedestrians in the provided image
    """

    # detect pedestrians
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    (rectangles, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(32, 32), scale=5)

    # Combine overlapping rectangles
    finalRectangles = non_max_suppression(np.array([[x, y, x + w, y + h] for (x, y, w, h) in rectangles]),
                                           overlapThresh=0.65)

    # Check to see if rectangles should be printed
    if drawRectangles:
        if len(rectangles):
            draw_rectangles(image, finalRectangles)

    return finalRectangles


def draw_rectangles(image, rectangles):
    """
    This function draws boxes on an image and displays the image.

    @:param image : a cv2 image with pedestrians
    @:param rectangles : a list of rectangles to draw on the image.
    @:return nothing but displays the image
    """
    imageCopy = image.copy()
    for (x0, y0, x1, y1) in rectangles:
        cv2.rectangle(imageCopy, (x0, y0), (x1, y1), (0, 0, 0), 2)

    plt.imshow(cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB))
    plt.show()

