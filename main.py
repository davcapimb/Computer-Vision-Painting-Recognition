import argparse
import os

from peopleDetection import inizializeModel
from pipeline import pipeline
from utils import *

# path management
paolo_path = 'D:/VCS-project/'
dav_path = '/media/davide/aukey/progetto_vision/'
pepp_path = '/media/peppepc/Volume/Peppe/Unimore/Vision and Cognitive Systems/Project material/'
dav_path_win = 'D:\progetto_vision/'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', type=str)
    parser.add_argument('--model', type=str, default='COCO')
    args = parser.parse_args()

    # -- PRELIMINAR OPERATIONS --#

    # the path containing all the project material
    root_path = args.root_path

    # painting database
    paintingsDB = loadPaintingsDB(root_path)

    # people detection model (options: COCO, PEDANT)
    model = inizializeModel(args.model)

    # videos
    videos_path = root_path + 'videos/'
    videos = []
    for root, dirs, files in os.walk(videos_path):
        for file in files:
            videos.append(str(root) + '/' + str(file))
    videos = np.random.permutation(videos)

    # museum map
    museumMap = cv2.imread(root_path + 'map.png')

    # -- VIDEO ANALYSIS"

    for videoFile in videos:
        video = cv2.VideoCapture(videoFile)
        print('WATCHING: ' + videoFile)
        frame_counter = 0

        while video.isOpened():
            _, frame = video.read()
            if frame is not None:
                frame_counter += 1
                if frame_counter % 5 == 0:
                    pipeline(frame, paintingsDB, model, museumMap)

                # next video
                if (cv2.waitKey(1) & 0xFF == ord('n')) or frame_counter == video.get(
                        cv2.CAP_PROP_FRAME_COUNT) - 1:
                    video.release()
                    break

    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
