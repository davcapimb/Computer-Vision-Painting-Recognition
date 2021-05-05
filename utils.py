import csv

import cv2
import numpy as np

framePadding = 60
cropPadding = framePadding - 5
innerBoxPadding = 100


# load the painting database
def loadPaintingsDB(root_path):
    print("Loading db..")
    # read csv file
    paintingsDB_path = root_path + 'paintings_db/'
    paintings_info_file = root_path + 'data.csv'
    with open(paintings_info_file) as file:
        paintings_info = list(csv.DictReader(file, delimiter=','))

    for paintFile in paintings_info:
        paint = cv2.imread(paintingsDB_path + paintFile['Image'])

        # compute the ORB descriptors for each painting
        if paint is not None:
            orb = cv2.ORB_create()
            paintKeypoints, paintDescriptors = orb.detectAndCompute(paint, None)
            paintFile['Desc'] = paintDescriptors
            paintFile['Painting'] = paint

    return paintings_info


# create a cropped frame containing only the bounding box with some padding
def crop(frame, bbox):
    bordered = cv2.copyMakeBorder(frame, framePadding, framePadding, framePadding, framePadding,
                                  borderType=cv2.BORDER_CONSTANT)
    x, y, w, h = bbox
    return bordered[y:y + h + 2 * cropPadding, x:x + w + 2 * cropPadding]


# remove the boxes contained in bigger ones
def removeInnerBox(peopleBoxes, paintingsBoxes):
    boxes = []
    errors = []
    if peopleBoxes is not None:
        [boxes.append(i) for i in peopleBoxes]
    if paintingsBoxes is not None:
        [boxes.append(i) for i in paintingsBoxes]

    if boxes is not None:
        for x11, y11, x12, y12 in boxes:
            for x21, y21, x22, y22 in boxes:
                if ((x11 != x21) & (y11 != y21) & (x12 != x22) & (y12 != y22)):
                    if ((x11 > x21 - innerBoxPadding) & (y11 > y21 - innerBoxPadding) & (
                            x12 < x22 + innerBoxPadding) & (y12 < y22 + innerBoxPadding)):
                        errors.append((x11, y11, x12, y12))
        if len(errors) > 0:
            if peopleBoxes is not None:
                peopleBoxes = [b for b in peopleBoxes if b not in errors]
            if paintingsBoxes is not None:
                paintingsBoxes = [b for b in paintingsBoxes if b not in errors]

        return peopleBoxes, paintingsBoxes


# change dimension of a picture to show that in a window
def resize(dim, img):
    r = dim / img.shape[1]
    dim = (dim, int(img.shape[0] * r))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


# standardize the order of the corners of the segmentated painting
def orderPoints(points):
    point = [('x', int), ('y', int)]
    values = [(points[0][0][0], points[0][0][1]),
              (points[1][0][0], points[1][0][1]),
              (points[2][0][0], points[2][0][1]),
              (points[3][0][0], points[3][0][1])]
    box = np.array(values, point)
    p0 = np.sort(np.sort(box, order='x')[:2], order='y')[0]
    p1 = np.sort(np.sort(box, order='x')[2:], order='y')[0]
    p2 = np.sort(np.sort(box, order='x')[2:], order='y')[-1]
    p3 = np.sort(np.sort(box, order='x')[:2], order='y')[-1]

    return np.int32([[[p0['x'], p0['y']]], [[p1['x'], p1['y']]], [[p2['x'], p2['y']]], [[p3['x'], p3['y']]]])
