import tensorflow as tf
import keras
import skimage
import cv2
import numpy as np
import mrcnn.visualize as vz

from skimage.transform import resize
from skimage.color import rgb2grey
import warnings



def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)




class TensorBoardImage(keras.callbacks.Callback):
    

    def __init__(self, tag, mrcnn_model, generator, config, logdir):
        super().__init__() 
        self.tag = tag        
        self.mrcnn_model = mrcnn_model
        self.generator = generator
        self.config = config
        #self.colors = visualize.random_colors(20)
        self.logdir = logdir
        self.writer = tf.summary.FileWriter(self.logdir)

    def detect(self, verbose=0):
        results = []


        for i in range(10):
            inputs, outputs = next(self.generator)           

            batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,\
                batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_inst_ids, batch_mold_image_meta, batch_mold_window = inputs        
            
            outputs = self.model.predict_on_batch([batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                                                    batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_inst_ids])
            mrcnn_class_logits, mrcnn_bbox, mrcnn_mask, detections,\
                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, inst_id_loss \
                = outputs
            
            
            for i, image in enumerate(batch_images):
                final_rois, final_class_ids, final_scores, final_masks =\
                    self.mrcnn_model.unmold_detections(detections[i], mrcnn_mask[i],
                                        image.shape, batch_images[i].shape,
                                        batch_mold_window[i])

                #inst_ids = instance_ids[i]
                results.append({
                    "image": self.mrcnn_model.unmold_image(image),
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                    #"inst_ids": inst_ids,

                })
        return results  


    def on_epoch_end(self, epoch, logs={}):
        
        results = self.detect()
        values = []
        for i, r in enumerate(results):
            


            im = vz.display_instances(r['image'], r['rois'], r['masks'], r['class_ids'], 
                                ["BG", "baloon"], r['scores'],
                                title="Predictions", auto_show = False)
          
            image = make_image(im)
            values.append(tf.Summary.Value(tag=self.tag + str(i), image=image))
        

        summary = tf.Summary(value=values)
        
        self.writer.add_summary(summary, epoch)
        self.writer.flush()
        #self.writer.close()

         
        print('Plot finished')

        return


class MRCNN_ModelCheckpoint(keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, backbone_filepath, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MRCNN_ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.backbone_filepath = backbone_filepath
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            backbone_filepath = self.backbone_filepath.format(epoch=epoch + 1, **logs)
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
    
                            self.model.get_layer("backbone_model").save_weights(backbone_filepath, overwrite=True)
                            self.model.get_layer("kira_model").save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.get_layer("backbone_model").save_weights(backbone_filepath, overwrite=True)
                    self.model.get_layer("kira_model").save_weights(filepath, overwrite=True)
                else:
                    self.model.get_layer("backbone_model").save(backbone_filepath, overwrite=True)
                    self.model.get_layer("kira_model").save(filepath, overwrite=True)