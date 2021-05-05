# Vision and Cognitive Systems final project


## Authors

- [Bertellini Paolo](https://github.com/paolobertellini)
- [Caputo Imbriaco Davide](https://github.com/davcapimb)
- [Doganieri Giuseppe](https://github.com/gdoganieri)

## Documentation
[Report paper on Overleef](https://www.overleaf.com/read/shnjmbddzjnf)
<br>
[Report presentation and paper](/documentation)

## Overview
The purpose of the project is to create a visual system capable to detect and recognize the paintings exhibited in ”Gallerie Estensi” museum of Modena and locating people that are visiting the museum understanding the room in which they are. The system in the future can be used by a robot for visiting the museum in an autonomous manner or can be installed on a mobile application to improve and guide the visitor’s experience.

![pipelineok](https://user-images.githubusercontent.com/45602824/111639290-68d0fd80-87fb-11eb-8682-a71a1823c7d1.png)

## Project highliths

The main aspects of the project’s pipeline are presented in this section. The motivations behind the choices made and the details of each task are discussed in the project documentation. 
The solution proposed takes as input a single frame from a video and passes it to the following pipeline:
#### 1. Detections
People and paintings are detected through two different approaches
  * People: given the large amount of data available on the web people are detected with a deep learning approach. In particular a faster RCNN pretrained on COCO dataset is used. The model returns a list of detection with the corresponding labels and bounding boxes. From the 80 COCO classes only the class “Person” is taken into account and all the others are discarded.
  * Paintings: the necessary amount of data to train a classiﬁer is not available, therefore an image processing approach is used. It is based on two main considerations:
    * The wall has a lighter color than the paintings;
    * Almost all the paintings are rectangular and the most of the circular ones have a rectangular frame.
Starting from this two considerations the model creates a mask ﬁltering the pixels of the wall that have hue between 80 and 255. The contours found in the mask are approximated
to polygons and only the ones with more than four side are taken into account.

<br> ![image](https://user-images.githubusercontent.com/45602824/111644883-88b6f000-8800-11eb-8526-b5676b8fedd0.png)
#### 2. Inner boxes removal
The boxes containing the paintings and the people detected are compared together in order to discard boxes that are contained in other boxes. This control allows the model to:
  * not consider people portrayed in the paintings;
  * consider only the outermost border if more than one is found for a single painting.

<br> ![image](https://user-images.githubusercontent.com/45602824/111644438-21993b80-8800-11eb-8d95-b3fc7d8bdcc8.png)
#### 3. Painting segmentation
Each painting detected and acknowledged by the inner box check is cropped from the frame with a padding and segmented using the Otsu threshold algorithm. The segmentation allows to detect the borders of the paintings with more precision and in particular to localize the four corner points that are crucial for the perspective rectiﬁcation.
<br> ![image](https://user-images.githubusercontent.com/45602824/111644533-38d82900-8800-11eb-8deb-ef157629e771.png)
#### 4. Painting rectiﬁcation
The four edge points B found with segmentation are used to compute the height H and the width W of the rectiﬁed version of the painting. Using W and H the model computes the new rectiﬁed box B0 with perpendicular corners. Then it warps the original frame using the transformation matrix obtained from the two set B and B0 and it crops the warped image in order to select just the painting region.
<br> ![image](https://user-images.githubusercontent.com/45602824/111643687-725c6480-87ff-11eb-8b40-592110adfd4b.png)
#### 5. Painting retrieval
For retrieval the information of the detected painting are computed the ORB descriptors on the crop portion of the frame and they are compared with the ones of all the paintings stored in the database. The function returns a sorted list of the number of strong matches found for each painting. The retrieval is considered reliable if there are almost 20 strong matches.
<br> ![image](https://user-images.githubusercontent.com/45602824/111644600-47264500-8800-11eb-80e8-cd35e95e4b38.png)
#### 6. People localization
The peopleLocalization combines the results of peopleDetection and paintingRetrieval. Whenever a person is detected the model is able to localize her retrieving the room from the database information of a painting recognized in the same frame of the person.
<br> ![image](https://user-images.githubusercontent.com/45602824/111644063-c1a29500-87ff-11eb-9043-f086d7003c19.png)

## How to use it

The program takes as input two parameters:
- root_path: the path of the google drive "Project Material" folder containing:
  * /videos             folder containing all the videos
  * /paintings_db       folder containing the painting database
  * data.csv            file with the informations of the painting database
  * map.png             image of the museum map
- model: the model to use for the people detection: there are actually two option available 'COCO' or 'PEDANT'

