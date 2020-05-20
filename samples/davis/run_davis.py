import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.transform
from skimage.transform import resize
from davis_data import io

from os import listdir, makedirs
from os.path import isfile, join, exists

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.davis import davis


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = davis.DavisConfig()

# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Load validation dataset
dataset = davis.DavisDataset()
dataset.load_davis("val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))


model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                        config=config)

run = 'simple_1080/davis20190701T1347'

epoch = 40 

log_folder = os.path.join('/home/fothar/cluster_segmentation/logs', run)

weights_path = {'backbone_model': os.path.join(log_folder, 'mask_rcnn_backbone_model_davis_{0:04d}.h5'.format(epoch)),
                    'kira_model': os.path.join(log_folder, 'mask_rcnn_kira_model_davis_{0:04d}.h5'.format(epoch))}


# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

result_root = '/home/fothar/DAVIS_results'

result_run = os.path.join(result_root, 'simple{}_{}_argmax'.format(run, epoch))

if not exists(result_run):
    makedirs(result_run)

for seq_id, seq_name in enumerate(dataset.db.iternames()):    
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    result_seq = os.path.join(result_run, seq_name)
    if not exists(result_seq):
        makedirs(result_seq)
    
    seq = dataset.db[seq_id]
    
    for frame_id in range(len(seq.images)):
        im = seq.images[frame_id]
        image , _, _, _, _ = utils.resize_image(
                im,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)
        
        # Run object detection
        results = model.detect([image], verbose=0)

        # Display results
        r = results[0]            

        inst_ids = r["inst_ids"]

        current_mask = np.argmax(inst_ids, axis=-1).astype(np.uint8)

        if frame_id == 0:

            mask = current_mask.copy()  
        else:
            inst_map = [0]
            for i in range(1, 6):
                overlap = current_mask[mask==i]
                if len(overlap):
                    current_id = np.argmax(np.bincount(overlap))                    
                    if current_id == 0:
                        current_id = i
                else:
                    current_id = i                    
                inst_map.append(current_id)                
            mask = np.zeros((inst_ids.shape[0], inst_ids.shape[1]), dtype=np.uint8)
            for prev_id, current_id in enumerate(inst_map):
                mask[current_mask==current_id]=prev_id       
    
        davis_out_file = '{:05d}.png'.format(frame_id)        
        davis_resized = resize(mask, (im.shape[1], im.shape[1]), order=0) * 255
        davis_resized = davis_resized.astype(np.uint8)
        padding = int((im.shape[1]- im.shape[0])/2)
        davis_resized = davis_resized[padding:-padding, :]
        io.imwrite_indexed(join(result_seq, davis_out_file), davis_resized)
    print(seq_name)
print(result_run)