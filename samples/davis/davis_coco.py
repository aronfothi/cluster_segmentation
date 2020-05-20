
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa
import imgaug as ia
from itertools import zip_longest

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config 
from mrcnn import siamese_model as modellib, utils

from davis_data import cfg, phase
from davis_data import io


from davis_data import DAVISLoader

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

coco_dataset_path = '/home/fothar/datasets/coco'

############################################################
#  Configurations
############################################################


class DavisConfig(Config):
    """Configuration for training on the DAVIS dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "davis"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + mask

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    NUM_INSTANCES = 6

    NUM_SAMPLES = 500

    LOSS_WEIGHTS = {
        "inst_id_loss": 1.,
        "bg_loss": 1.,


        "rpn_class_loss_A": 1.,
        "rpn_bbox_loss_A": 1.,
        "mrcnn_class_loss_A": 1.,
        "mrcnn_bbox_loss_A": 1.,
        "mrcnn_mask_loss_A": 1.,
        "inst_id_loss_A": 1.,
        "bg_loss_A": 1.,

        "rpn_class_loss_B": 1.,
        "rpn_bbox_loss_B": 1.,
        "mrcnn_class_loss_B": 1.,
        "mrcnn_bbox_loss_B": 1.,
        "mrcnn_mask_loss_B": 1.,
        "inst_id_loss_B": 1.,
        "bg_loss_B": 1.,

        "siamese_inst_id_loss": 2.
    }


############################################################
#  Dataset
############################################################
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

DEFAULT_DATASET_YEAR = "2014"
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

class DavisDataset(utils.Dataset):

    def load_davis(self, subset):
        """Load a subset of the DAVIS dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("davis", 1, "davis")

        # Train or validation dataset?
        assert subset in ["train", "val"] 
        
        height = 2160
        width = 3840
        self.db = DAVISLoader(year="2017",phase=phase[subset.upper()])
        seq_names = self.db.iternames()
        for seq_id, seq_name in enumerate(seq_names):
            seq = self.db[seq_id]
            mask = seq.annotations[0]
            unq = len(np.unique(mask))
            for frame_id in range(len(seq.images)):
                self.add_image(
                    "davis",
                    path='dummy',
                    image_id='{}_{}'.format(str(seq_id), str(frame_id)),
                    frame_id=frame_id,
                    seq=seq,
                    unq=unq)

    

    def load_image(self, image_id):
        info = self.image_info[image_id]
        return info["seq"].images[info["frame_id"]]

    def load_all_with_pair(self, image_id, select_next=False):
        info = self.image_info[image_id]
        selected = int(info["frame_id"])
        if select_next:
            r = selected + 1
        else:
            n_frames = len(info["seq"].images)
            n_frames -= 1
            n, p = n_frames, selected/n_frames
            r = np.random.binomial(n, p)

        image_a = info["seq"].images[info["frame_id"]]
        image_b = info["seq"].images[r]

        annotations = info["seq"].annotations

        mask_a = annotations[info["frame_id"]]
        mask_b = annotations[r]
        bool_mask_a = np.zeros((mask_a.shape[0], mask_a.shape[1], annotations.n_objects), dtype=np.bool)
        bool_mask_b = np.zeros((mask_b.shape[0], mask_b.shape[1], annotations.n_objects), dtype=np.bool)
        for i, oi in enumerate(annotations.iter_objects_id()):
            bool_mask_a[:, :, i] = mask_a == oi
            bool_mask_b[:, :, i] = mask_b == oi

        class_ids_a = np.ones([bool_mask_a.shape[-1]], dtype=np.int32)
        class_ids_b = np.ones([bool_mask_b.shape[-1]], dtype=np.int32)
        return image_a, image_b, bool_mask_a, bool_mask_b, class_ids_a, class_ids_b


    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info["seq"].annotations
        mask = annotations[info["frame_id"]]
        bool_mask = np.zeros((mask.shape[0], mask.shape[1], annotations.n_objects), dtype=np.bool)
        for i, oi in enumerate(annotations.iter_objects_id()):
            bool_mask[:, :, i] = mask == oi
        
        return bool_mask, np.ones([bool_mask.shape[-1]], dtype=np.int32)

    
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train_davis = DavisDataset()
    dataset_train_davis.load_davis("train")
    dataset_train_davis.prepare()
 
    dataset_train_coco = CocoDataset()
    dataset_train_coco.load_coco(coco_dataset_path, "train")
    dataset_train_coco.load_coco(coco_dataset_path, "valminusminival")
    dataset_train_coco.prepare()

    # Validation dataset
    dataset_val = DavisDataset()
    dataset_val.load_davis("val")
    dataset_val.prepare()

    augmentation = iaa.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train([dataset_train_davis, dataset_train_coco], dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)           

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train([dataset_train_davis, dataset_train_coco], dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+',
                augmentation=augmentation)





############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect DAVIS masks.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    
    args = parser.parse_args()

    # Validate arguments
    #if args.command == "train":
        

    print("Weights: ", args.weights)
    #print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    DEVICE = "/gpu:1"
    #import tensorflow as tf
    #with tf.device(DEVICE):

    # Configurations
    if args.command == "train":
        config = DavisConfig()
    else:
        class InferenceConfig(DavisConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = {'backbone_model': model.get_imagenet_weights()}
    elif args.weights.lower() == "custom":
        # Start from ImageNet trained weights
        weights_path = {'backbone_model': '/home/fothar/cluster_segmentation_redesign/kira_custom/mask_rcnn_backbone_model_coco.h5',
                        'kira_model': '/home/fothar/cluster_segmentation_redesign/kira_custom/mask_rcnn_kira_model_coco.h5'}
    else:
        weights_path = args.weights

    # Load weights
    print("#######################Loading weights ", weights_path)
    if args.weights.lower() == "coco" or args.weights.lower() == "custom":
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
    else:
        print("'{}' is not recognized. "
            "Use 'train' or 'splash'".format(args.command))
