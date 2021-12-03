import cv2
import matplotlib.pyplot as plt


def extract_features(image, showImage):
    """
    This function extracts features of a detected pedestrian.

    @:param image : a cv2 image of a pedestrian
    @:param showImage : a boolean value to show the feature extraction
    @:return image keypoints
    """

    sift = cv2.SIFT_create()
    kp1, temp = sift.detectAndCompute(image, None)

    if showImage:
        imageCopy = image.copy()
        imageCopy = cv2.drawKeypoints(imageCopy, kp1, imageCopy)
        plt.imshow(imageCopy)
        plt.show()

    return kp1
