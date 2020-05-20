from davis_data import io
import numpy as np
from skimage.transform import resize
import cv2

from os import listdir, makedirs
from os.path import isfile, join, exists

root_path = '/home/fothar/cluster_segmentation_redesign/samples/davis'
numpy_result = join(root_path, 'numpyarr_result', 'kira_only')
davis_result = join(root_path, 'davis_result')
np_files = [f for f in listdir(numpy_result) if isfile(join(numpy_result, f))]

davis_mask_path = '/home/fothar/davis-2017/data/DAVIS/Annotations/480p/'

for np_file in np_files:
    np_data = np.load(join(numpy_result, np_file))
    scene_id = np_file.split("_")[-2]

    davis_out_folder = join(davis_result, scene_id)
    if not exists(davis_out_folder):
        makedirs(davis_out_folder)

    scene_inst_ids = set()

    for i in range(np_data.shape[0]):
        davis_out_file = '{:05d}.png'.format(i)
        gt_path = join(davis_mask_path, scene_id, davis_out_file)
        gt_mask, _ = io.imread_indexed(gt_path)

        scene_inst_ids = scene_inst_ids | set(np.unique(gt_mask)[1:])

    scene_inst_ids = list(scene_inst_ids)

    mask_inst_ids = np.unique(np_data)[1:]

    id_map = []

    for gt_id in scene_inst_ids:
        max_mask_id = 0
        max_mask = 0

        for mask_id in mask_inst_ids:
            summa = 0

            for i in range(np_data.shape[0]):
                davis_out_file = '{:05d}.png'.format(i)
                gt_path = join(davis_mask_path, scene_id, davis_out_file)
                gt_mask, _ = io.imread_indexed(gt_path)

                davis_resized = resize(np_data[i], (gt_mask.shape[1], gt_mask.shape[1]), order=0) *255
                davis_resized = davis_resized.astype(np.uint8)

                padding = int((gt_mask.shape[1]- gt_mask.shape[0])/2)

                davis_resized = davis_resized[padding:-padding, :]


                summa += np.sum((davis_resized==mask_id) == (gt_mask==gt_id))

            if summa>max_mask:
                max_mask = summa
                max_mask_id = mask_id
        
        id_map.append(max_mask_id)

    old_np_data = np_data.copy()
    for i, gt_id in enumerate(scene_inst_ids):
        np_data[old_np_data==id_map[i]] = gt_id

    
    for i in range(np_data.shape[0]):
        davis_out_file = '{:05d}.png'.format(i)
        gt_path = join(davis_mask_path, scene_id, davis_out_file)
        gt_mask, _ = io.imread_indexed(gt_path)
        
        davis_resized = resize(np_data[i], (gt_mask.shape[1], gt_mask.shape[1]), order=0) *255
        davis_resized = davis_resized.astype(np.uint8)

        padding = int((gt_mask.shape[1]- gt_mask.shape[0])/2)

        davis_resized = davis_resized[padding:-padding, :]


        
        frame = np.concatenate((gt_mask*70, davis_resized*70))

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

        io.imwrite_indexed(join(davis_out_folder, davis_out_file), davis_resized)

    davis_out_file = '{:05d}.png'.format(i+1)
    io.imwrite_indexed(join(davis_out_folder, davis_out_file), davis_resized)
        
    print(scene_id)
    