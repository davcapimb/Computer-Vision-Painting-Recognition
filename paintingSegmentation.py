import cv2
import numpy as np

from utils import orderPoints


def paintingSegmentation(cropped):
    # gray scale convertion and noise clean up
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    cropped = cv2.GaussianBlur(cropped, (5, 5), 0)

    # Otsu thresholding
    _, th = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th

    # contours research
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    minArea = np.size(cropped)
    bestSegmentation = None

    for cnt in contours:
        # contour poligon approximation
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # quadrilateral filtering
        if len(approx) == 4:
            polygonArea = cv2.contourArea(approx)
            if polygonArea > (np.size(cropped) / 3) / 10:
                # polygon area filtering
                if polygonArea < minArea:
                    bestSegmentation = orderPoints(approx)
                    minArea = polygonArea

    return bestSegmentation
