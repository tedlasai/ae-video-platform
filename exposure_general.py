import math
import numpy as np
import scipy
import collections
import constants


class Exposure:
    def __init__(self,
                raw_images,
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
        self.out_img_width = 168  # default downsized width
        self.out_img_height = 112  # default downsized height
        self.width = None
        self.height = None
        self.absolute_bit = 2 ** 8  # max bit number of the raw image
        self.raw_images = raw_images
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
        self.initial_functions()

    # helper function to add two 4d arrays those might have different shape in 3red and 4th dimrntions(trim thr larger one)

    def initial_functions(self):
        self.compute_downsample_size()

    def compute_downsample_size(self):
        self.num_frame, self.num_ims_per_frame, self.height, self.width = self.raw_images.shape
        if 0 < self.downsample_rate <= 1:
            one_d_down_rate = math.sqrt(self.downsample_rate)
            self.out_img_height = int(self.height * one_d_down_rate)
            self.out_img_width = int(self.width * one_d_down_rate)

    def adjusted_opti_inds_v2_by_average_of_previous_n_frames(self, opti_inds):
        length = len(opti_inds)
        opti_inds_new = np.array(opti_inds)
        # print("# previes frames")
        # print(opti_inds)
        # print(self.number_of_previous_frames)
        # print("step size")
        # print(self.stepsize)
        current_index = opti_inds_new[0]
        # if length > 1:
        #     i = 1
        #     while i < length:
        #         start_index = max(0, i - self.number_of_previous_frames)
        #         # print("start_ind: "+str(start_index))
        #         average_of_previous_n_frames = np.mean(opti_inds_new[start_index:i])
        #         diff = average_of_previous_n_frames - opti_inds_new[i]
        #         if diff < -self.stepsize:
        #             opti_inds_new[i] = round(average_of_previous_n_frames + self.stepsize)
        #         if diff > self.stepsize:
        #             opti_inds_new[i] = round(average_of_previous_n_frames - self.stepsize)
        #         i += 1
        init_list = np.ones(self.number_of_previous_frames)*opti_inds[0]
        #lastVisitedIndices = collections.deque([opti_inds[0],opti_inds[0],opti_inds[0]], maxlen=3)
        lastVisitedIndices = collections.deque(init_list, maxlen=self.number_of_previous_frames)

        if length > 2:
            i = 1
            while i < length:
                # print("start_ind: "+str(start_index))
                sum = 0
                for index in lastVisitedIndices:
                    sum+=index
                average_of_previous_n_frames = sum/self.number_of_previous_frames
                #print(f"i{i}, average_of_previous_n_frames {average_of_previous_n_frames},  opti_inds_new[i] { opti_inds_new[i]}")
                if(abs(average_of_previous_n_frames - opti_inds_new[i] )> 1):
                   # print(lastVisitedIndices)
                    nextVisitIndex = (lastVisitedIndices[-2] + 2*lastVisitedIndices[-1] +3*opti_inds_new[i])/6
                    #current_index = opti_inds_new[i]
                    opti_inds_new[i] = nextVisitIndex
                else:
                    opti_inds_new[i] = opti_inds_new[i-1]
                current_index = opti_inds_new[i]
                lastVisitedIndices.append(current_index)



                #override change
                #if abs(opti_inds_new[i]-opti_inds_new[i-1]) < 3 and( opti_inds_new[i-1]-opti_inds_new[i-2]) < 2:
                #    opti_inds_new[i] = opti_inds_new[i-1]
                i += 1
        print("OPT NEW", opti_inds_new)
        return opti_inds_new
