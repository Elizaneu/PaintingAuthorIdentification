import cv2
from skimage.filters import sato
from skimage.filters import sobel as sob


# sobel feature
def sobel(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = sob(img)
    return result, result


# sato feature
def Sato(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = sato(img)
    return result, result


# fast feature
def fast(image):
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(50)
    kp = fast.detect(image, None)
    result = cv2.drawKeypoints(image, kp, None, color=(255, 0, 0))
    return len(kp), result
