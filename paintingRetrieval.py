import cv2


def paintingRetrieval(painting, paintings_info):
    # compute descriptors of the detected painting
    orb = cv2.ORB_create()
    _, paintingDescriptors = orb.detectAndCompute(painting, None)

    paintingsScores = []

    for i, pd in enumerate(paintings_info):
        # compute distances using Hamming
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        # find the two nearest descriptors
        matches = bf.knnMatch(paintingDescriptors, pd['Desc'], k=2)

        strongMatches = []

        # filter the strongest descriptors
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                strongMatches.append(m.distance)
        # save the results into a list
        ps = {'index': i, 'n': len(strongMatches)}
        paintingsScores.append(ps)

    # sort list by descending similarity
    paintingsScores = sorted(paintingsScores, key=lambda l: l['n'], reverse=True)

    # initialize the information to return
    info = paintings_info[paintingsScores[0]['index']]['Title'][:10] + '(' + str(
        paintingsScores[0]['index']) + ')' + '[' + str(paintingsScores[0]['n']) + ']'
    retrievedPainting = None  # cv2.imread('notfound.png')
    room = 'UNDEFINED'

    # retrieval confidence control
    if paintingsScores[0]['n'] >= 20:
        boxColor = (0, 255, 0)
        retrievedPainting = paintings_info[paintingsScores[0]['index']]['Painting']
        room = paintings_info[paintingsScores[0]['index']]['Room']

    elif 5 < paintingsScores[0]['n'] < 20:
        boxColor = (0, 255, 255)

    else:
        boxColor = (0, 0, 255)

    return paintingsScores, boxColor, info, room, retrievedPainting
