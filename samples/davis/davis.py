
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from davis_data import cfg, phase
from davis_data import io


from davis_data import DAVISLoader

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs", 'simple_480')

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
    VALIDATION_STEPS = 50

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    NUM_INSTANCES = 6

    NUM_SAMPLES = 100

    LEARNING_RATE=0.0005    

    TRAIN_BN = False

    LOSS_WEIGHTS = {
        "inst_id_loss": 2.,
        "bg_loss": 10.,
    }
    BACKBONE = "resnet50"
    '''
    LOSS_WEIGHTS = {
        "inst_id_loss": 1.,
        "bg_loss": 10.,


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
    }'''


############################################################
#  Dataset
############################################################

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
    dataset_train = DavisDataset()
    dataset_train.load_davis("train")
    #dataset_train.load_davis("val")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DavisDataset()
    dataset_val.load_davis("val")
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=augmentation)           

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=180,
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
        weights_path = {'backbone_model': '/home/fothar/cluster_segmentation/logs/coco20190627T1441/mask_rcnn_backbone_model_coco.h5',
                        'kira_model': '/home/fothar/cluster_segmentation/logs/coco20190627T1441/mask_rcnn_kira_model_coco.h5'}
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
