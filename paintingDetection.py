import cv2


def paintingDetection(frame):
    # HSV conversion and noise cleanup
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.medianBlur(frame_hsv, 25)

    # wall pixel filtering
    mask = cv2.inRange(frame_hsv, (0, 0, 80), (255, 255, 255))
    mask = 255 - mask

    # contours research
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxesDetected = []

    for i, cnt in enumerate(contours):
        # contour poligon approximation
        epsilon = 0.05 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # polygon filtering
        if cv2.contourArea(approx) > 15000:
            if len(approx) >= 4:
                # rectangle approximation
                bbox = cv2.boundingRect(approx)
                x, y, w, h = bbox
                boxesDetected.append((x, y, x + w, y + h))

    return boxesDetected
