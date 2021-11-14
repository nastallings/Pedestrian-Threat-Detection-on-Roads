import cv2
import matplotlib.pyplot as plt


def extract_features(image, showImage):
    """
    This function extracts features of a detected pedestrian.

    @:param image : a cv2 image of a pedestrian
    @:param showImage : a boolean value to show the feature extraction
    @:return image keypoints
    """

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, temp = sift.detectAndCompute(image, None)

    if showImage:
        image = cv2.drawKeypoints(image, kp1, image)
        plt.imshow(image)
        plt.show()

    return kp1
