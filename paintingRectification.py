import math

import cv2
import numpy as np


def perspectiveRectification(frame, segmentedPoints):
    # rectified painting height
    cardH = math.sqrt((segmentedPoints[2][0][0] - segmentedPoints[1][0][0]) * (
            segmentedPoints[2][0][0] - segmentedPoints[1][0][0]) +
                      (segmentedPoints[2][0][1] - segmentedPoints[1][0][1]) * (
                              segmentedPoints[2][0][1] - segmentedPoints[1][0][1]))
    # rectified painting width
    cardW = math.sqrt((segmentedPoints[0][0][0] - segmentedPoints[1][0][0]) * (
            segmentedPoints[0][0][0] - segmentedPoints[1][0][0]) +
                      (segmentedPoints[0][0][1] - segmentedPoints[1][0][1]) * (
                              segmentedPoints[0][0][1] - segmentedPoints[1][0][1]))
    # segmented box
    box = np.float32([[segmentedPoints[0][0][0], segmentedPoints[0][0][1]],
                      [segmentedPoints[1][0][0], segmentedPoints[1][0][1]],
                      [segmentedPoints[2][0][0], segmentedPoints[2][0][1]],
                      [segmentedPoints[3][0][0], segmentedPoints[3][0][1]]])
    # rectified box
    rectBox = np.float32([[segmentedPoints[0][0][0], segmentedPoints[0][0][1]],
                          [segmentedPoints[0][0][0] + cardW, segmentedPoints[0][0][1]],
                          [segmentedPoints[0][0][0] + cardW, segmentedPoints[0][0][1] + cardH],
                          [segmentedPoints[0][0][0], segmentedPoints[0][0][1] + cardH]])

    # transformation matrix (from segmented to rectified)
    M = cv2.getPerspectiveTransform(box, rectBox)

    offsetSize = 400
    blackFrame = np.zeros((int(cardW + offsetSize), int(cardH + offsetSize)), dtype=np.uint8);

    # warp frame using the transformation matrix
    warpedFrame = cv2.warpPerspective(frame, M, blackFrame.shape)
    # crop painting region
    rectifiedPainting = warpedFrame[int(rectBox[0][1]):int(rectBox[2][1]), int(rectBox[0][0]):int(rectBox[1][0])]

    return rectifiedPainting
