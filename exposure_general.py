import math
import numpy as np
import scipy

import constants


class Exposure:
    def __init__(self,
                input_images,
                downsample_rate=1 / 64,
                r_percent=0,
                g_percent=1,
                high_threshold=1,
                low_threshold=0,
                high_rate=0.2,
                low_rate=0.2,
                num_hist_bins=100,
                stepsize=3,
                number_of_previous_frames=5,
                start_index = 20,):
        #self.global_rate = global_rate
        self.absolute_bit = 2 ** 8  # max bit number of the raw image
        self.input_images = input_images
        self.downsample_rate = downsample_rate  # down sample rate of the original images, preferred value is as # 1/perfectsquare (i.e 1/36, 1/81)
        self.g_percent = g_percent  # weight of green channel
        if r_percent + g_percent > 1:
            self.r_percent = 1 - g_percent
        else:
            self.r_percent = r_percent  # weight of red channel
        self.b_percent = 1 - r_percent - g_percent
        # self.col_num_grids = col_num_grids  # number of grids in x axis
        # self.row_num_grids = row_num_grids
        self.low_threshold = low_threshold  # low outlier threshold
        self.high_threshold = high_threshold
        self.high_rate = high_rate  # down sample rate of the areas over high outlier threshold
        self.low_rate = low_rate
        # self.local_indices = local_indices  # a list of 4d coordinates [0:self.num_frame,0:num_ims_per_frame,0:y_num_grids,0:col_num_grids], which indicates the interested grids
        # self.grid_h = 0  # hight of a grid
        # self.grid_w = 0
        self.num_frame = 0
        self.num_ims_per_frame = 0
        self.h = 0  # hight of a downsampled image
        self.w = 0
        # self.target_intensity = target_intensity
        self.num_hist_bins = num_hist_bins
        # self.local_with_downsampled_outliers = local_with_downsampled_outliers  # a flag indicates if it should downsample the outlier areas when such area is the local interested area or not.("True" means it should downsample the outliers)
        self.stepsize = stepsize
        self.number_of_previous_frames = number_of_previous_frames
        self.SCALE_LABELS = constants.SCALES
        self.indexes_out_of_40 = constants.indexes_out_of_40
        self.NEW_SCALES = constants.NEW_SCALES
        self.start_index = start_index

    # helper function to add two 4d arrays those might have different shape in 3red and 4th dimrntions(trim thr larger one)