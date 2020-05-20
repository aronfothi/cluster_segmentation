import os
import sys
import random
import math
import numpy as np
import cv2

import colorsys

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from samples.davis import davis


def hsv2rgb(h,s,v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: hsv2rgb(*c), hsv))
    random.shuffle(colors)
    return colors

config = davis.DavisConfig()
# Load validation dataset
dataset = davis.DavisDataset()
dataset.load_davis("val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

colors = random_colors(6)
colors = colors*255


for seq_id, seq_name in enumerate(dataset.db.iternames()):    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    vid_name = seq_name + '_val_4.mp4'

    mrcnn_path = '/home/fothar/cluster_segmentation/samples/davis/mrcnn_val_100_' + seq_name + '_90.npy'
    simple_path = '/home/fothar/cluster_segmentation_redesign/samples/davis/kira_simple_val_100_' + seq_name + '_85.npy'
    siamese_path = '/home/fothar/cluster_segmentation_redesign/samples/davis/kira_only_val_100_' + seq_name + '_85.npy'
    early_simple_path = '/home/fothar/cluster_segmentation_redesign/samples/davis/early_kira_simple_val_100_' + seq_name + '_90.npy'
    early_siamese_path = '/home/fothar/cluster_segmentation_redesign/samples/davis/early_kira_only_val_100_' + seq_name + '_90.npy'

    mrcnn_mask = np.load(mrcnn_path)
    simple_mask = np.load(simple_path)
    siamese_mask = np.load(siamese_path)
    early_simple_mask = np.load(early_simple_path)
    early_siamese_mask = np.load(early_siamese_path)
    
    #out = cv2.VideoWriter(vid_name,fourcc, 20.0, (768, 512))
    out = cv2.VideoWriter(vid_name,fourcc, 20.0, (768, 512))
    seq = dataset.db[seq_id]
    for frame_id in range(len(seq.images)-1):
        im_a = seq.images[frame_id]
        im_a , _, _, _, _ = utils.resize_image(
                im_a,
                min_dim=config.IMAGE_MIN_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                max_dim=config.IMAGE_MAX_DIM,
                mode=config.IMAGE_RESIZE_MODE)

        img_mrcnn = im_a.copy()
        img_simple = im_a.copy()
        img_siamese = im_a.copy()
        early_img_simple = im_a.copy()
        early_img_siamese = im_a.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_mrcnn,'MRCNN',(10,50), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img_simple,'Simple',(10,50), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img_siamese,'Siamese',(10,50), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(early_img_simple,'early_Simple',(10,50), font, 2,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(early_img_siamese,'early_Siamese',(10,50), font, 2,(255,255,255),2,cv2.LINE_AA)

        for i in range(1, 6):

            img_mrcnn[mrcnn_mask[frame_id]==i] = colors[i]
            img_simple[simple_mask[frame_id]==i] = colors[i]
            img_siamese[siamese_mask[frame_id]==i] = colors[i]
            early_img_simple[early_simple_mask[frame_id]==i] = colors[i]
            early_img_siamese[early_siamese_mask[frame_id]==i] = colors[i]

        new_shape = (256, 256)
        im_a = cv2.resize(im_a[:, :, ::-1], new_shape)
        img_mrcnn = cv2.resize(img_mrcnn[:, :, ::-1], new_shape)
        img_simple = cv2.resize(img_simple[:, :, ::-1], new_shape)
        img_siamese = cv2.resize(img_siamese[:, :, ::-1], new_shape)
        early_img_simple = cv2.resize(early_img_simple[:, :, ::-1], new_shape)
        early_img_siamese = cv2.resize(early_img_siamese[:, :, ::-1], new_shape)

        first_row = np.concatenate((im_a, img_mrcnn))
        second_row = np.concatenate((img_simple, img_siamese))
        third_row = np.concatenate((early_img_simple, early_img_siamese))
        frame = np.concatenate((first_row, second_row, third_row), axis=1)
        out.write(frame)
        
    out.release()
    print(vid_name)