# Mask R-CNN trained model for Detecting images of cars damaged in an accident
# This is based off the Mask R-CNN implementation by matterport and shared as public git repository //github.com/matterport/Mask_RCNN

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

## Citation
Use this bibtex to cite this repository:
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```
## Installation
Prerequisite - you should have python 3.x installed

1. Clone this repository and change to working directory
   ```cd Machine-Learning/automatic_car_accident_detection_using_mask_rnn```
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository 
    ```bash
    python3 setup.py install
    ```
4. Download pre-trained COCO weights (mask_rcnn_coco.h5) from https://github.com/matterport/Mask_RCNN/releases/tag/v2.0 if you want to train something yourself else you can download the pretrained weights for the car accident images as given in next step

5. Download pre-trained 50 epoch weights (mask_rcnn_caraccident_0050.h5) for car accident from http://tiny.cc/RCNNmodel

## To Run

1. After downloading the source code
2. Download the pretrained car accident weights "mask_rcnn_caraccident_0050.h5" from step 5 above and copy them under automatic_car_accident_detection_using_mask_rnn/carAccidentDetect directory
3. execute python3 carAccidentDetecy.py --command="test" --weights="./mask_rcnn_caraccident_0050.h5" --image=<fully qualified path and name of image>
