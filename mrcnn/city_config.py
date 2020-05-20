from config import Config

class CityConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "CityConfig"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 3#5 #8

    # Number of classes (including background)
    NUM_CLASSES = 20
    NUM_INSTANCES = 8

    NUM_SAMPLES = 50
    NUM_PAIRS = 10000

    EPSILON = 256


    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
  

    LEARNING_RATE = 0.01
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    RESNET = 101

    FEATURE_MAP_SIZE = 512

    CITYDATA_PATH = '/home/fothar/cityscapes/'

