# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

__author__ = 'federico perazzi'
__version__ = '2.0.0'

from davis_data.misc import log     # Logger
from davis_data.misc import cfg     # Configuration parameters
from davis_data.misc import phase   # Dataset working set (train,val,etc...)
from davis_data.misc import overlay # Overlay segmentation on top of RGB image
from davis_data.misc import Timer   # Timing utility class
from davis_data.misc import io

from davis_data.dataset import DAVISLoader,Segmentation,Annotation


