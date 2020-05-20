import utils
import os
import sys
import numpy as np

from skimage import io


class CityDataset(utils.Dataset):

    def load_instances(self, config, subset):
        """Load a subset of the cityscapes dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset lane, and the class lane
        self.add_class("instance", 1, "instance")
        
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(config.CITYDATA_PATH)
        drive_image_dir = os.path.join(dataset_dir, 'leftImg8bit', subset)

        no_instance = ['dusseldorf_000101_000019',
                        'dusseldorf_000106_000019',
                        'weimar_000097_000019',
                        'weimar_000067_000019',
                        'strasbourg_000000_035571',
                        'strasbourg_000000_012934',
                        'strasbourg_000000_036016',
                        'strasbourg_000000_023854',
                        'bochum_000000_031152',
                        'monchengladbach_000000_015561',
                        'lindau_000049_000019',
                        'lindau_000019_000019',
                        'lindau_000017_000019',
                        'lindau_000018_000019',
                        'lindau_000021_000019',
                        'lindau_000032_000019',
                        'lindau_000040_000019',
                        'lindau_000045_000019',
                        
                        ]

        drive_ids = os.listdir(drive_image_dir)

        for drive_id in drive_ids:

            image_dir = os.path.join(drive_image_dir, drive_id)
            mask_dir  = os.path.join(dataset_dir, 'gtFine', subset, drive_id)
            sp_dir    = os.path.join(dataset_dir, 'superpixel', subset, drive_id)

            image_ids = ['_'.join(f.split('_')[:-1] ) for f in os.listdir(image_dir)]            
            # Add images
            for image_id in image_ids:
                
                if image_id not in no_instance:
                    '''
                    mask = cv2.imread(os.path.join(mask_dir, "{}_gtFine_instanceIds.png".format(image_id)), 3)
                    crop_size = (640, 640)
                    mask = self.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)

                    instance_ids = np.unique(mask)
                    instance_ids = [i for i in instance_ids if i >=1000]
                    if(len(instance_ids)<1):
                        print('sdsdasdasdasdasdasdsadasdasdas',image_id )
                    '''
                    self.add_image(
                        "instance",
                        image_id=image_id,
                        path=os.path.join(image_dir, "{}_leftImg8bit.png".format(image_id)),
                        mask_path=os.path.join(mask_dir, "{}_gtFine_instanceIds.png".format(image_id)),
                        label_path=os.path.join(mask_dir, "{}_gtFine_labelIds.png".format(image_id)),
                        train_path=os.path.join(mask_dir, "{}_gtFine_labelTrainIds.png".format(image_id)),
                        color_path=os.path.join(mask_dir, "{}_gtFine_color.png".format(image_id)),
                        sp_path = os.path.join(sp_dir, "{}_leftImg8bit.png".format(image_id)),
                        orig_id=image_id)

    def load_image(self, image_id):

        img = io.imread(self.image_info[image_id]['path']) 
        
        return img 

    def load_superpixel(self, image_id):

        img = io.imread(self.image_info[image_id]['sp_path']) 
        
        return img        
                

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a lane dataset image, delegate to parent class.
        mask=cv2.imread(self.image_info[image_id]['mask_path'], 3)
        
        return mask

        

    def load_mask_test(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a lane dataset image, delegate to parent class.
        inst=io.imread(self.image_info[image_id]['mask_path']).astype(np.uint8)
        color = io.imread(self.image_info[image_id]['mask_path']).astype(np.uint8)
        label = io.imread(self.image_info[image_id]['label_path']).astype(np.uint8)
        sem = io.imread(self.image_info[image_id]['train_path']).astype(np.uint8)
        return inst, color, label, sem
    

    def get_path(self, image_id):
        return  self.image_info[image_id]['path'], self.image_info[image_id]['mask_path']

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "instance":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

