"""

First Write : Vanshaj Vohra Aug 2018
Last Write : Vanshaj Vohra Jul 2019

The aim of this project is to detect cars that have suffered accidental damage from a given image.
This is based on Mask R-CNN object detection and instance segmentation on Keras and TensorFlow models put by Matterport in public domain.  https://github.com/matterport/Mask_RCNN

Usage :

    # Train a new model starting from pre-trained COCO weights
    python3 carAccidentDetect.py train --dataset=<directory with 2 subdirectories named 'train' and 'val'> --weights=<trained weights>
    python3 carAccidentDetect.py test --weights=<path to trained weights file.h5> --image=<path to test image file>
    python3 carAccidentDetect.py splash --weights=<path to trained weights file.h5> --video=<path to test video file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
#from mrcnn.visualize import display_instances
#import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
foundMatch=0
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
class_names = ['BG', 'caraccident']
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
GENERATED_IMG_DIR = "./dataset/generated/"

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "carAccident"

    # I just used my macbook pro and so defined 1 image per GPU
    # But in case you have a high end machine with Nvidia GPU and
    # 100 G RAM, up scale to 2 to 3 image for faster processing
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):

        # Add classes. We have only one class to add.
        self.add_class("carAccident", 1, "carAccident")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "carAccident",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        if image_info["source"] != "carAccident":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "carAccident":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model. Only Train heads"""
    print("Training Started..... this will take some time")
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')



def detect_and_draw_box(model, image_path=None, video_path=None):
    assert image_path or video_path
    global foundMatch
    global fileCount
    # Image or video?
    if image_path:
        fileCount+=1
        import cv2
        #print("Running on {}".format(args.image))
        # Read image
        image = cv2.imread(image_path)
        #image = skimage.io.imread(image_path)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        #print(r['rois'])
        #print(r['masks'])
        #print(r['scores'])
        if r['rois'].size > 0:
            foundMatch+=1
            for (y1,x1,y2,x2) in r['rois']:
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
            #cv2.imshow("frame",image)
            #cv2.waitKey(1000) & 0xFF == ord('q')
            file_name = "./dataset/generated/accidentFound/carAccDetect_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            cv2.imwrite(file_name, image)
        else:
            file_name = "./dataset/generated/accidentNotFound/carAccDetect_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            cv2.imwrite(file_name,image)
        #cv2.destroyAllWindows()
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        print("Video FPS is ")
        print(fps)

        # Define codec and create video writer
        file_name = "./dataset/generated/carAccDetect_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:

                # Detect objects
                r = model.detect([image], verbose=0)[0]
                #print(r['scores'])
                #print(r['rois'])
                #if not r['rois']:
                if r['rois'].size > 0:
                   time = count/fps
                   print("Accident detected")
                   print(time)
                   for (y1,x1,y2,x2) in r['rois']:
                       cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)

                vwriter.write(image)
                count += 1
        vwriter.release()
    #print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect images with car in accident state.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.video,\
               "Provide --image or --video to test for an image of a car in accident"
    elif args.command == "bulktest":
        assert args.image or args.video, \
        "Provide --image or --video to test for an image of a car in accident"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)


    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "test":
        fileCount=0
        foundMatch=0
        detect_and_draw_box(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "bulktest":
        imageDir=args.image
        foundMatch=0
        fileCount=0
        import os
        files = os.listdir(imageDir)
        for fname in files:
            fullImageName = os.path.join(imageDir, fname)
            print("Working on ",fullImageName)
            print(fileCount)
            print(foundMatch)
            detect_and_draw_box(model, image_path=fullImageName)
        print(fileCount)
        print(foundMatch)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'test'".format(args.command))

