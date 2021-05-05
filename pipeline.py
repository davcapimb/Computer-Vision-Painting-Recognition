from paintingDetection import paintingDetection
from paintingRectification import perspectiveRectification
from paintingRetrieval import paintingRetrieval
from paintingSegmentation import paintingSegmentation
from peopleDetection import peopleDetection
from utils import *

# room boxes localization on the museum map
rooms = {'1': [(932, 405), (1035, 605)], '2': [(632, 605), (1035, 705)], '3': [(830, 605), (930, 705)],
         '4': [(725, 605), (830, 705)],
         '5': [(620, 605), (725, 705)], '6': [(514, 605), (620, 705)], '7': [(413, 605), (514, 705)],
         '8': [(357, 605), (413, 705)],
         '9': [(261, 605), (357, 705)], '10': [(211, 605), (261, 705)], '11': [(110, 605), (211, 705)],
         '12': [(10, 605), (110, 705)],
         '13': [(10, 405), (110, 605)], '14': [(10, 305), (110, 405)], '15': [(10, 110), (10, 305)],
         '16': [(10, 10), (145, 110)],
         '17': [(145, 10), (220, 110)], '18': [(10, 220), (310, 110)], '19': [(110, 110), (310, 405)],
         '20': [(110, 405), (414, 605)],
         '21': [(414, 405), (722, 605)], '22': [(722, 405), (933, 605)]}

# output windows management
cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow("Detection", 10, 10)
cv2.resizeWindow("Detection", 800, 500)

cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
cv2.moveWindow("Segmentation", 10, 545)
cv2.resizeWindow("Segmentation", 350, 250)

cv2.namedWindow('Rectification', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Rectification", 810, 10)

cv2.namedWindow('Retrieval', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("Retrieval", 1180, 10)

cv2.namedWindow('People localization', cv2.WINDOW_NORMAL)
cv2.moveWindow("People localization", 380, 545)


def pipeline(frame, paintings_info, model, mapMuseum):
    # clean copy of the frame
    peopleDet_frame = frame.copy()
    segmentation_frame = frame.copy()
    detection_frame = frame.copy()

    # inizialize people localization info
    museumMap = mapMuseum.copy()
    room = 'UNDEFINED'

    # red copy of the frame
    red_frame = np.full(frame.shape, (0, 0, 255), np.uint8)
    segmentation_frame = cv2.addWeighted(segmentation_frame, 0.4, red_frame, 0.6, 0)

    # people detection
    peopleBoxes = peopleDetection(frame, model, 0.8)

    # painting detection
    paintingsBoxes = paintingDetection(frame)

    # inner box removal
    if (peopleBoxes is not None) | (paintingsBoxes is not None):
        peopleBoxes, paintingsBoxes = removeInnerBox(peopleBoxes, paintingsBoxes)

    # draw bounding box around people
    if peopleBoxes is not None:
        for x1, y1, x2, y2 in peopleBoxes:
            cv2.rectangle(peopleDet_frame, (x1, y1), (x2, y2), (255, 255, 255), 5)

    for x1, y1, x2, y2 in paintingsBoxes:
        boxColor = (0, 0, 255)
        bbox = [x1, y1, x2 - x1, y2 - y1]

        # cut the detected painting
        croppedPainting = crop(frame, bbox)

        # painting segmentation
        paintingSegmented = paintingSegmentation(croppedPainting)

        if paintingSegmented is not None:

            # draw segmented painting
            cv2.drawContours(segmentation_frame, [paintingSegmented + [bbox[0] - 60, bbox[1] - 60]], 0, (0, 255, 0), -1)

            # painting rectification
            rect_painting = perspectiveRectification(croppedPainting, paintingSegmented)
            if np.size(rect_painting) != 0:
                cv2.imshow("Rectification", resize(350, rect_painting))

        # painting retrieval
        paintingScore, boxColor, paintingInfo, room, retrievalImage = paintingRetrieval(croppedPainting, paintings_info)
        cv2.putText(detection_frame, paintingInfo, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        if retrievalImage is not None:
            cv2.imshow("Retrieval", resize(350, retrievalImage))

        # draw painting box
        cv2.rectangle(detection_frame, (x1, y1), (x2, y2), boxColor, 5)

    # draw people box
    if peopleBoxes is not None:
        for x1, y1, x2, y2 in peopleBoxes:
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
            cv2.putText(detection_frame, 'people in room ' + room, (x1, y1 - 15), cv2.FONT_HERSHEY_DUPLEX, 1,
                        (255, 255, 255), 2)
        # people localization
        if room != 'UNDEFINED':
            r = np.full(frame.shape, (0, 255, 0), np.uint8)
            cv2.rectangle(museumMap, rooms[room][0], rooms[room][1], (0, 255, 0), -1)

    cv2.imshow('Detection', detection_frame)
    cv2.imshow('Segmentation', segmentation_frame)
    cv2.imshow('People localization', museumMap)
