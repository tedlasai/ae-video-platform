import math
import numpy as np
import scipy


class Exposure:
    def __init__(self, input_images, downsample_rate=1 / 64, r_percent=0, g_percent=1, col_num_grids=8, row_num_grids=8,
                 target_intensity=0.18, high_threshold=1, low_threshold=0, high_rate=0.2, low_rate=0.2,
                 local_indices=[], num_hist_bins=100, local_with_downsampled_outliers=False, stepsize=3,
                 number_of_previous_frames=5, global_rate=0, start_index = 20):
        self.global_rate = global_rate
        self.absolute_bit = 2 ** 8  # max bit number of the raw image
        self.input_images = input_images
        self.downsample_rate = downsample_rate  # down sample rate of the original images, preferred value is as # 1/perfectsquare (i.e 1/36, 1/81)
        self.g_percent = g_percent  # weight of green channel
        if r_percent + g_percent > 1:
            self.r_percent = 1 - g_percent
        else:
            self.r_percent = r_percent  # weight of red channel
        self.b_percent = 1 - r_percent - g_percent
        self.col_num_grids = col_num_grids  # number of grids in x axis
        self.row_num_grids = row_num_grids
        self.low_threshold = low_threshold  # low outlier threshold
        self.high_threshold = high_threshold
        self.high_rate = high_rate  # down sample rate of the areas over high outlier threshold
        self.low_rate = low_rate
        self.local_indices = local_indices  # a list of 4d coordinates [0:self.num_frame,0:num_ims_per_frame,0:y_num_grids,0:col_num_grids], which indicates the interested grids
        self.grid_h = 0  # hight of a grid
        self.grid_w = 0
        self.num_frame = 0
        self.num_ims_per_frame = 0
        self.h = 0  # hight of a downsampled image
        self.w = 0
        self.target_intensity = target_intensity
        self.num_hist_bins = num_hist_bins
        self.local_with_downsampled_outliers = local_with_downsampled_outliers  # a flag indicates if it should downsample the outlier areas when such area is the local interested area or not.("True" means it should downsample the outliers)
        self.stepsize = stepsize
        self.number_of_previous_frames = number_of_previous_frames
        self.SCALE_LABELS = [15, 8, 6, 4, 2, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 15, 1 / 30, 1 / 60, 1 / 125, 1 / 250, 1 / 500]
        self.indexes_out_of_40 = [0, 3, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
        self.NEW_SCALES = [15,13,10,8,6,5,4,3.2,2.5,2,1.6,1.3,1,0.8,0.6,0.5,0.4,0.3,1/4,1/5,1/6,1/8,1/10,1/13,1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100,1/125,1/160,1/200,1/250,1/320,1/400,1/500]
        self.start_index = start_index

    # helper function to add two 4d arrays those might have different shape in 3red and 4th dimrntions(trim thr larger one)
    def add_two_4d_arrays(self, x, y):
        c_x = x.shape[2]
        c_y = y.shape[2]
        diff = c_y - c_x
        if diff > 0:
            y = y[:, :, :-diff, :]
        elif diff < 0:
            x = x[:, :, :diff, :]
        d_x = x.shape[3]
        d_y = y.shape[3]
        diff = d_y - d_x
        if diff > 0:
            y = y[:, :, :, :-diff]
        elif diff < 0:
            x = x[:, :, :, :diff]
        return x + y

    def set_absolute_bit(self, matrix):
        max_val = (np.max(matrix))
        log2_ = math.ceil(math.log2(max_val))
        # self.absolute_bit = 2**log2_
        self.absolute_bit = 2 ** 8

    # helper function to downsample one channel
    def downsample_one_channel(self, matrix, channel, row_skip_step, col_skip_step):
        if channel == 'g1':
            return matrix[:, :, ::row_skip_step, 1::col_skip_step] / 2 / (self.absolute_bit - 1)
        if channel == 'g2':
            return matrix[:, :, 1::row_skip_step, ::col_skip_step] / 2 / (self.absolute_bit - 1)
        if channel == 'r':
            return matrix[:, :, ::row_skip_step, ::col_skip_step] / (self.absolute_bit - 1)
        return matrix[:, :, 1::row_skip_step, 1::col_skip_step] / (self.absolute_bit - 1)

    def downsample_blending_rgb_channels(self):
        if self.downsample_rate >= 0.75:
            self.downsample_rate = 1
        elif self.downsample_rate > 0.5:
            self.downsample_rate = 0.5
        sqrt_rate = math.sqrt(1 / self.downsample_rate)
        decimal_ = sqrt_rate - int(sqrt_rate)
        if decimal_ > 0.85 or decimal_ < 0.15:
            col_skip_step = round(sqrt_rate) * 2
            row_skip_step = col_skip_step
        else:
            col_skip_step = math.ceil(sqrt_rate) * 2
            row_skip_step = math.floor(sqrt_rate) * 2

        raw_bayer = np.load(self.input_images)
        # print("input shape")
        # print(raw_bayer.shape)
        self.num_frame, self.num_ims_per_frame, orig_h, orig_w = raw_bayer.shape
        self.set_absolute_bit(raw_bayer)
        # when cfa is rggb
        # green channel [0,1] [1,0]
        # red channel [0,0]
        # blue channel [1,1]
        r = []
        b = []
        g = []
        if self.r_percent > 0:
            r = self.downsample_one_channel(raw_bayer, 'r', row_skip_step, col_skip_step)
        if self.b_percent > 0:
            b = self.downsample_one_channel(raw_bayer, 'b', row_skip_step, col_skip_step)
        if self.g_percent > 0:
            g1 = self.downsample_one_channel(raw_bayer, 'g1', row_skip_step, col_skip_step)
            g2 = self.downsample_one_channel(raw_bayer, 'g2', row_skip_step, col_skip_step)
            g = self.add_two_4d_arrays(g1, g2)

        if self.r_percent == 0:
            if self.b_percent == 0:  # green only
                result = g
            elif self.g_percent == 0:  # blue only
                result = b
            else:  # green + blue
                g = g * self.g_percent
                b = b * self.b_percent
                result = self.add_two_4d_arrays(g, b)
        elif self.b_percent == 0:
            if self.g_percent == 0:  # red only
                result = r
            else:  # red + green
                g = g * self.g_percent
                r = r * self.r_percent
                result = self.add_two_4d_arrays(g, r)
        elif self.g_percent == 0:  # red + blue
            r = r * self.r_percent
            b = b * self.b_percent
            result = self.add_two_4d_arrays(r, b)
        else:  # red + blue + green
            r = r * self.r_percent
            b = b * self.b_percent
            r_b = self.add_two_4d_arrays(r, b)
            g = g * self.g_percent
            result = self.add_two_4d_arrays(r_b, g)
        self.num_frame, self.num_ims_per_frame, self.h, self.w = result.shape
        self.set_grid_h_w()
        return result

    def set_grid_h_w(self):
        self.grid_h = int(self.h / self.row_num_grids)
        self.grid_w = int(self.w / self.col_num_grids)

    # the function returns 2 arrays (the partition indexs of x axis & the partion indexs of y axis)
    # 0 excluded and h,w included
    # the grid h/w is rounded down, the extra rows/cols will be dropped later
    def partition_boundaries(self):
        partition_boundaries_col = np.arange(0, self.w, self.grid_w)[:self.col_num_grids]
        partition_boundaries_row = np.arange(0, self.h, self.grid_h)[:self.row_num_grids]
        return partition_boundaries_col, partition_boundaries_row

    def get_grided_intesntiy(self, imgs):
        partition_boundaries_col, partition_boundaries_row = self.partition_boundaries()
        grid_means = ([np.mean(imgs[x, y, j:j + self.grid_h, i:i + self.grid_w]) for x in range(self.num_frame) for y in
                       range(self.num_ims_per_frame) for j in partition_boundaries_row for i in
                       partition_boundaries_col])
        grid_means = np.reshape(grid_means,
                                (self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
        return grid_means

    def get_grided_intesntiy_from_grided_ims(self, grided_imgs):
        temp_ims = np.mean(grided_imgs, axis=4)
        temp_ims = np.mean(temp_ims, axis=3)
        grid_means = np.reshape(temp_ims,
                                (self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
        return grid_means

    def get_grided_ims(self, imgs):
        num_frame, num_ims_per_frame, h, w = imgs.shape
        self.set_grid_h_w()
        partition_boundaries_col, partition_boundaries_row = self.partition_boundaries()
        grid_ims = [np.array(imgs[x, y, j:j + self.grid_h, i:i + self.grid_w]) for x in range(self.num_frame) for y in
                    range(self.num_ims_per_frame) for j in partition_boundaries_row for i in partition_boundaries_col]
        grid_ims = np.array(grid_ims)
        grid_ims = np.reshape(grid_ims, (
            num_frame, num_ims_per_frame, self.row_num_grids * self.col_num_grids, self.grid_h, self.grid_w))
        grid_means = self.get_grided_intesntiy_from_grided_ims(grid_ims)
        return grid_ims, grid_means

    # assume the matrix is in the shape of (num_frame, num_ims_per_frame,row_num_grids,col_num_grids)
    def fourd_indices_to_oned_indices(self, fourd_indices):
        oned_indices = []
        for ind in fourd_indices:
            oned_ind = ind[0] * self.num_ims_per_frame * self.row_num_grids * self.col_num_grids + ind[
                1] * self.row_num_grids * self.col_num_grids + ind[2] * self.col_num_grids + ind[3]
            oned_indices.append(oned_ind)
        return oned_indices

    def get_lists_of_outlier_one_d_indices(self, grid_means):
        high_indices = np.where(grid_means > self.high_threshold)
        high_coord_list = list(zip(high_indices[0], high_indices[1], high_indices[2], high_indices[3]))
        high_oned_indices = self.fourd_indices_to_oned_indices(high_coord_list)
        low_indices = np.where(grid_means < self.low_threshold)
        low_coord_list = list(zip(low_indices[0], low_indices[1], low_indices[2], low_indices[3]))
        low_oned_indices = self.fourd_indices_to_oned_indices(low_coord_list)

        return high_oned_indices, low_oned_indices

    # def get_grids_weight_matrix(self, grid_means):
    #     if len(self.local_indices) != 0:
    #         oned_indices = self.fourd_indices_to_oned_indices(self.local_indices)
    #         weights = np.zeros((self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
    #         weights.flat[oned_indices] = 1
    #         if self.local_with_downsampled_outliers:
    #             high_oned_indices, low_oned_indices = self.get_lists_of_outlier_one_d_indices(grid_means)
    #             high_oned_indices = list(filter(lambda x: x in oned_indices, high_oned_indices))
    #             low_oned_indices = list(filter(lambda x: x in oned_indices, low_oned_indices))
    #             print("high indices ")
    #             print(high_oned_indices)
    #             print("low indices")
    #             print(low_oned_indices)
    #         else:
    #             return weights
    #
    #     else:
    #         weights = np.ones((self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
    #         high_oned_indices, low_oned_indices = self.get_lists_of_outlier_one_d_indices(grid_means)
    #     weights.flat[high_oned_indices] = self.high_rate
    #     weights.flat[low_oned_indices] = self.low_rate
    #     return weights

    def get_grids_weight_matrix(self, grid_means):
        if len(self.local_indices) != 0:
            oned_indices = self.fourd_indices_to_oned_indices(self.local_indices)
            weights = np.zeros((self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
            weights.flat[oned_indices] = 1.0
            if self.local_with_downsampled_outliers:
                high_indices = np.where(grid_means > self.high_threshold)
                low_indices = np.where(grid_means < self.low_threshold)
            else:
                return weights, weights

        else:
            weights = np.ones((self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
            high_indices = np.where(grid_means > self.high_threshold)
            low_indices = np.where(grid_means < self.low_threshold)
        weights_before = np.array(weights)
        weights[high_indices] *= self.high_rate
        weights[low_indices] *= self.low_rate
        return weights, weights_before

    # helper function to convert (row_num_grids,col_num_grids) to an one d index
    def twod_indices_to_oned_index(self, twod_ind):
        oned_ind = twod_ind[0] * self.col_num_grids + twod_ind[1]
        return oned_ind

    def get_flatten_weighted_imgs(self, weights, grid_ims):
        num_of_pixels_per_grid = self.grid_w * self.grid_h
        flatten_weighted_ims = np.ones((self.num_frame, self.num_ims_per_frame, self.h * self.w)) * (-0.01)
        for (i, j, k, l), weight in np.ndenumerate(weights):
            third_ind_of_grided_ims = self.twod_indices_to_oned_index([k, l])
            if 0 < weight < 1:
                flatten_list = grid_ims[i, j, third_ind_of_grided_ims].flatten()
                sampled_num = int(num_of_pixels_per_grid * weight)
                flatten_weighted_im_per_grid = np.random.choice(flatten_list, sampled_num, replace=False)
                start = third_ind_of_grided_ims * num_of_pixels_per_grid
                end = start + sampled_num
                flatten_weighted_ims[i][j][start:end] = flatten_weighted_im_per_grid

            elif weight == 1:
                flatten_weighted_im_per_grid = grid_ims[i, j, third_ind_of_grided_ims].flatten()
                start = third_ind_of_grided_ims * num_of_pixels_per_grid
                end = (1 + third_ind_of_grided_ims) * num_of_pixels_per_grid
                flatten_weighted_ims[i][j][start:end] = flatten_weighted_im_per_grid
        return flatten_weighted_ims

    def get_flatten_weighted_imgs_local_wo_grids_moving_object(self, ims):
        if len(self.local_indices) == 0:
            flatten_weighted_ims = ims
        else:
            # flatten_weighted_ims = np.ones((self.num_frame, self.num_ims_per_frame, self.h , self.w)) * (-0.01)
            total_pixels = self.num_frame * self.num_ims_per_frame * self.h * self.w
            flatten_weighted_ims = np.ones(total_pixels) * (-0.01)
            if self.global_rate > 0:
                inds = np.random.choice(total_pixels, int(total_pixels * self.global_rate), replace=False)
                flatten_weighted_ims[inds] = 2
            flatten_weighted_ims = flatten_weighted_ims.reshape(
                (self.num_frame, self.num_ims_per_frame, self.h, self.w))
            for i in range(self.num_frame):
                for (y_start, x_start, y_end, x_end) in self.local_indices[i]:
                    y_start = int(y_start * self.h)
                    x_start = int(x_start * self.w)
                    y_end = int(y_end * self.h)
                    x_end = int(x_end * self.w)
                    flatten_weighted_ims[i, :, y_start:y_end + 1, x_start:x_end + 1] = ims[i, :, y_start:y_end + 1,
                                                                                       x_start:x_end + 1]
        if self.global_rate > 0:
            ind2d = np.where(flatten_weighted_ims == 2)
            flatten_weighted_ims[ind2d] = ims[ind2d]
        flatten_weighted_ims = flatten_weighted_ims.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))
        flatten_weighted_ims_before_outlier = np.array(flatten_weighted_ims)
        flatten_weighted_ims[flatten_weighted_ims < self.low_threshold] = -0.01
        flatten_weighted_ims[flatten_weighted_ims > self.high_threshold] = -0.01

        return flatten_weighted_ims, flatten_weighted_ims_before_outlier

    def get_flatten_weighted_imgs_local_wo_grids_moving_object_v2(self, ims):
        if len(self.local_indices) == 0:
            local_area = ims
        else:
            local_area = np.ones((self.num_frame, self.num_ims_per_frame, self.h, self.w)) * (-0.01)
            for i in range(self.num_frame):
                for (y_start, x_start, y_end, x_end) in self.local_indices[i]:
                    y_start = int(y_start * self.h)
                    x_start = int(x_start * self.w)
                    y_end = int(y_end * self.h)
                    x_end = int(x_end * self.w)
                    local_area[i, :, y_start:y_end + 1, x_start:x_end + 1] = ims[i, :, y_start:y_end + 1,
                                                                             x_start:x_end + 1]
        if self.global_rate > 0:
            global_area = np.where(local_area == -0.01, ims, -0.01)
        else:
            global_area = np.ones((self.num_frame, self.num_ims_per_frame, self.h, self.w)) * (-0.01)
        local_area = local_area.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))
        local_area_before_outlier = np.array(local_area)
        local_area[local_area < self.low_threshold] = -0.01
        local_area[local_area > self.high_threshold] = -0.01

        global_area = global_area.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))
        global_area_before_outlier = np.array(global_area)
        global_area[global_area < self.low_threshold] = -0.01
        global_area[global_area > self.high_threshold] = -0.01

        return local_area, local_area_before_outlier, global_area, global_area_before_outlier

    # the following version randomly take a percentage of global areas
    def get_flatten_weighted_imgs_local_wo_grids(self, ims):
        # flatten_weighted_ims = np.ones((self.num_frame, self.num_ims_per_frame, self.h , self.w)) * (-0.01)
        total_pixels = self.num_frame * self.num_ims_per_frame * self.h * self.w
        flatten_weighted_ims = np.ones(total_pixels) * (-0.01)
        if self.global_rate > 0:
            inds = np.random.choice(total_pixels, int(total_pixels * self.global_rate), replace=False)
            flatten_weighted_ims[inds] = 2
        flatten_weighted_ims = flatten_weighted_ims.reshape((self.num_frame, self.num_ims_per_frame, self.h, self.w))
        for (y_start, x_start, y_end, x_end) in self.local_indices:
            y_start = int(y_start * self.h)
            x_start = int(x_start * self.w)
            y_end = int(y_end * self.h)
            x_end = int(x_end * self.w)
            flatten_weighted_ims[:, :, y_start:y_end + 1, x_start:x_end + 1] = ims[:, :, y_start:y_end + 1,
                                                                               x_start:x_end + 1]
        if self.global_rate > 0:
            ind2d = np.where(flatten_weighted_ims == 2)
            flatten_weighted_ims[ind2d] = ims[ind2d]
        flatten_weighted_ims = flatten_weighted_ims.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))
        flatten_weighted_ims_before_outlier = np.array(flatten_weighted_ims)
        flatten_weighted_ims[flatten_weighted_ims < self.low_threshold] = -0.01
        flatten_weighted_ims[flatten_weighted_ims > self.high_threshold] = -0.01

        return flatten_weighted_ims, flatten_weighted_ims_before_outlier

    def get_hists(self, flatten_weighted_ims):
        scene_hists_include_drooped_counts = self.hist_laxis(flatten_weighted_ims, self.num_hist_bins + 2, (
            -0.01, 1.01))  # 2 extra bin is used to count the number of -0.01 and 1
        num_dropped_pixels = scene_hists_include_drooped_counts[:, :, 0]
        scene_hists = scene_hists_include_drooped_counts[:, :, 1:]
        return scene_hists, num_dropped_pixels

    def hist_laxis(self, data, n_bins,
                   range_limits):  # https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
        # Setup bins and determine the bin location for each element for the bins
        R = range_limits
        N = data.shape[-1]
        bins = np.linspace(R[0], R[1], n_bins + 1)
        data2D = data.reshape(-1, N)
        idx = np.searchsorted(bins, data2D, 'right') - 1

        # Some elements would be off limits, so get a mask for those
        bad_mask = (idx == -1) | (idx == n_bins)

        # We need to use bincount to get bin based counts. To have unique IDs for
        # each row and not get confused by the ones from other rows, we need to
        # offset each row by a scale (using row length for this).
        scaled_idx = n_bins * np.arange(data2D.shape[0])[:, None] + idx

        # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
        limit = n_bins * data2D.shape[0]
        scaled_idx[bad_mask] = limit

        # Get the counts and reshape to multi-dim
        counts = np.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
        counts.shape = data.shape[:-1] + (n_bins,)
        return counts

    def correct_one_mean(self, input_):
        mean, num_drooped_pixels_per_im = input_
        img_size = self.h * self.w
        if img_size == num_drooped_pixels_per_im:  # if all the pixels are drooped, return -1
            return -1
        return (mean * img_size - (-0.01) * num_drooped_pixels_per_im) / (img_size - num_drooped_pixels_per_im)

    def get_means(self, num_dropped_pixels, flatten_weighted_ims):
        #weighted_all_means = np.mean(flatten_weighted_ims, axis=2).reshape((self.num_frame, self.num_ims_per_frame, 1))
        #num_dropped_pixels = num_dropped_pixels.reshape((self.num_frame, self.num_ims_per_frame, 1))
        #concat_weighted_means_num_dropped_pixels = np.concatenate((weighted_all_means, num_dropped_pixels), axis=2)
        #weighted_means_flatten = np.apply_along_axis(self.correct_one_mean, 2, concat_weighted_means_num_dropped_pixels)
        #weighted_means = weighted_means_flatten.reshape(self.num_frame, self.num_ims_per_frame)

        num_good_pixels = flatten_weighted_ims != -0.01
        num_good_pixels = np.sum(num_good_pixels, axis=2)
        num_good_pixels[num_good_pixels == 0] = -1 # this allows for division when values are 0(usally really bright stuff)
        good_pixel_ims = flatten_weighted_ims
        good_pixel_ims[good_pixel_ims < 0] = 0 #make all the negative 0.01s to 0

        weighted_means = np.sum(good_pixel_ims, axis=2) / num_good_pixels #take average of good pixels

        return weighted_means

    def get_optimal_img_index(self, weighted_means):
        abs_weighted_errs_between_means_target = np.abs(weighted_means - self.target_intensity)
        return np.argmin(abs_weighted_errs_between_means_target, axis=1)

    def adjusted_opti_inds(self, opti_inds, stepsize=3):
        length = len(opti_inds)
        opti_inds_new = np.array(opti_inds)
        if length > 1:
            for i in range(1, length):
                diff = opti_inds_new[i - 1] - opti_inds_new[i]
                if diff < -stepsize:
                    opti_inds_new[i] = opti_inds_new[i - 1] + stepsize
                if diff > stepsize:
                    opti_inds_new[i] = opti_inds_new[i - 1] - stepsize
        return opti_inds_new

    def adjusted_opti_inds_v2_by_average_of_previous_n_frames(self, opti_inds):
        length = len(opti_inds)
        opti_inds_new = np.array(opti_inds).astype(int)
        print("# previes frames")
        print(opti_inds)
        print(self.number_of_previous_frames)
        print("step size")
        print(self.stepsize)
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
        import collections
        lastVisitedIndices = collections.deque([opti_inds[0],opti_inds[0],opti_inds[0],opti_inds[0]], maxlen=4)

        if length > 2:
            i = 1
            while i < length:
                # print("start_ind: "+str(start_index))
                sum = 0
                for index in lastVisitedIndices:
                    sum+=index
                average_of_previous_n_frames = sum/4
                #print(f"i{i}, average_of_previous_n_frames {average_of_previous_n_frames},  opti_inds_new[i] { opti_inds_new[i]}")
                if(abs(average_of_previous_n_frames - opti_inds_new[i] )> 0.5):
                   # print(lastVisitedIndices)
                    nextVisitIndex = round((lastVisitedIndices[-3]+lastVisitedIndices[-2] + 2*lastVisitedIndices[-1] +3*opti_inds_new[i])/6)
                    print("TYPE NEXT VISIT", type(nextVisitIndex), nextVisitIndex)
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
        #opti_inds_new = 15*np.ones(length).astype(int)
        return opti_inds_new

    def HDR_weight_function(self, x):
        if x <= 0.5:
            return 2 * x
        else:
            return 2 - 2 * x

    def build_HDR_imgs(self):
    #    SCALE_LABELS = [15, 8, 6, 4, 2, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 15, 1 / 30, 1 / 60, 1 / 125, 1 / 250, 1 / 500]
    #     indexes_out_of_40 = [0, 3, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
        downsampled_ims_ = self.downsample_blending_rgb_channels()
        downsampled_ims = downsampled_ims_[:, self.indexes_out_of_40, :, :]
        shutter_speed_reciprocal = 1 / np.array(self.SCALE_LABELS)
        weight_matrix = np.where(downsampled_ims <= 0.5, 2 * downsampled_ims, 2 - 2 * downsampled_ims)
        weighted_ims = np.multiply(weight_matrix, downsampled_ims)
        weighted_ims = np.multiply(weighted_ims, shutter_speed_reciprocal[None, :, None, None])
        weighted_ims_sum = np.sum(weighted_ims, axis=1)
        weight_matrix_sum = np.sum(weight_matrix, axis=1)
        weight_matrix_sum = weight_matrix_sum + 0.00001
        weight_matrix_sum_reciprocal = 1 / weight_matrix_sum
        hdr_ims = np.multiply(weight_matrix_sum_reciprocal, weighted_ims_sum)
        return hdr_ims, downsampled_ims_

    def get_max_area_exposure_time(self,hdr_ims):
        black_level = 0.03  # 511.7 get from the black image captured
        white_level = 0.85   # to be changed
        new_scales_reciprocal = 1 / np.array(self.NEW_SCALES)
        self.minHDR = np.ones(len(self.NEW_SCALES)) * black_level
        self.maxHDR = np.ones(len(self.NEW_SCALES)) * white_level
        print(self.maxHDR)
        print("***")
        self.minHDR = np.multiply(self.minHDR, new_scales_reciprocal)
        self.maxHDR = np.multiply(self.maxHDR, new_scales_reciprocal)
        print(self.maxHDR)
        x, y, z = hdr_ims.shape
        ims = hdr_ims.reshape(x, y * z)
        hdr_slot_sums = np.apply_along_axis(self.get_max_area_exposure_time_one_flatten_frame, 1, ims)
        result = (np.argmax(hdr_slot_sums,axis=1)).reshape(len(hdr_slot_sums))
        return result

    def hdr_max_area_pipeline(self):
        hdr_ims, downsampled_ims = self.build_HDR_imgs()
        opti_inds = self.get_max_area_exposure_time(hdr_ims)
        print(type(opti_inds))
        print(opti_inds)

        grided_ims, grided_means = self.get_grided_ims(downsampled_ims)
        weights, weights_before_ds_outlier = self.get_grids_weight_matrix(grided_means)
        flatten_weighted_ims = self.get_flatten_weighted_imgs(weights, grided_ims)
        flatten_weighted_ims_before_ds_outlier = self.get_flatten_weighted_imgs(weights_before_ds_outlier, grided_ims)
        hists, dropped = self.get_hists(flatten_weighted_ims)
        hists_before_ds_outlier, dropped_before_ds_outlier = self.get_hists(flatten_weighted_ims_before_ds_outlier)
        weighted_means = self.get_means(dropped, flatten_weighted_ims)

        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier


    def get_max_area_exposure_time_one_flatten_frame(self,im):
        result = []
        for i in range(len(self.minHDR)):
            result.append([])
            max_ = self.maxHDR[i]
            min_ = self.minHDR[i]
            # in_ranged = np.where(im <= max_, im, -0.1)
            # in_ranged = np.where(in_ranged >= min_, in_ranged, -0.1)
            # count = len(np.where(in_ranged >= 0))
            count = np.count_nonzero((im >= min_) & (im <= max_))
            #result[i].append(np.sum(in_ranged))
            result[i].append(count)
        return np.array(result)







    # To do: add reference paper of this method
    def gradient_srgb_exposure_pipeline(self):
        # downsampled_ims = self.downsample_blending_rgb_channels()
        # downsampled_ims = downsampled_ims**(1/3.4)\
        raw_bayer = np.load(self.input_images)
        raw_bayer = raw_bayer[:,:,::8,::8,:]
        current_rgb_img = raw_bayer / (2**8 - 1)
        current_rgb_img[:, :, :, :, 0] = current_rgb_img[:, :, :, :, 0] * 0.2126
        current_rgb_img[:, :, :, :, 1] = current_rgb_img[:, :, :, :, 1] * 0.7152
        current_rgb_img[:, :, :, :, 2] = current_rgb_img[:, :, :, :, 2] * 0.0722
        downsampled_ims = np.sum(current_rgb_img, axis=4)
        lam = 1000
        sig = 0.06
        dh = scipy.ndimage.sobel(downsampled_ims, axis=2)  # height
        dw = scipy.ndimage.sobel(downsampled_ims, axis=3)  # width
        d = np.hypot(dh, dw)
        maxes = np.amax(d, axis=(2, 3))
        d = d / 0.125  # normalize
        N = np.log(lam * (1 - sig) + 1)
        d = np.where(d < sig, sig, d)
        #dm = np.where(d <= sig, 0, (1 / N) * np.log((d - sig) * lam + 1))
        dm = (1 / N) * np.log((d - sig) * lam + 1)
        M = np.sum(dm, axis=(2, 3))
        opti_inds = np.argmax(M, axis=1)

        # grided_ims, grided_means = self.get_grided_ims(downsampled_ims)
        # weights, weights_before_ds_outlier = self.get_grids_weight_matrix(grided_means)
        # flatten_weighted_ims = self.get_flatten_weighted_imgs(weights, grided_ims)
        # flatten_weighted_ims_before_ds_outlier = self.get_flatten_weighted_imgs(weights_before_ds_outlier, grided_ims)
        # hists, dropped = self.get_hists(flatten_weighted_ims)
        # hists_before_ds_outlier, dropped_before_ds_outlier = self.get_hists(flatten_weighted_ims_before_ds_outlier)
        # weighted_means = self.get_means(dropped, flatten_weighted_ims)
        # opti_inds_mean_approach = self.get_optimal_img_index(weighted_means)
        # print(opti_inds)
        # print("gradient------mean")
        # print(opti_inds_mean_approach)
        weighted_means = np.zeros((100,40))
        hists = np.zeros((100,40,101))
        hists_before_ds_outlier = np.zeros((100,40,101))
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier

    def entropy_pipeline(self):

        raw_bayer = np.load(self.input_images)
        raw_bayer = raw_bayer[:, :, ::8, ::8, :]
        current_rgb_img = raw_bayer / (2 ** 8 - 1)
        current_rgb_img[:, :, :, :, 0] = current_rgb_img[:, :, :, :, 0] * 0.2126
        current_rgb_img[:, :, :, :, 1] = current_rgb_img[:, :, :, :, 1] * 0.7152
        current_rgb_img[:, :, :, :, 2] = current_rgb_img[:, :, :, :, 2] * 0.0722
        downsampled_ims = np.sum(current_rgb_img, axis=4)

        num_frames, stack_size, height, width = downsampled_ims.shape
        downsampled_ims1 = np.reshape(downsampled_ims, (num_frames, stack_size, height * width))
        opti_inds = []
        ind = self.start_index
        opti_inds.append(ind)
        from skimage.measure import shannon_entropy
        for j in range(1, 100):
            current_frame = downsampled_ims1[j]
            current_weighted_ims = []

            # current_map = current_map-0.10392
            entropies = np.empty(40)
            for i in range(40):
                entropies[i] = shannon_entropy(current_frame[i])

            ind = np.argmax(entropies)

            opti_inds.append(ind)

        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)

        weighted_means = np.zeros((100, 40))
        hists = np.zeros((100, 40, 101))
        hists_before_ds_outlier = np.zeros((100, 40, 101))

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier

    def pipeline(self):
        downsampled_ims = self.downsample_blending_rgb_channels()
        #generate histograms
        hist_ims = np.array(downsampled_ims)
        hist_ims[hist_ims>self.high_threshold] = -0.01
        hist_ims = np.reshape(hist_ims, (self.num_frame, self.num_ims_per_frame, self.h * self.w))
        hists, dropped = self.get_hists(hist_ims)
        weighted_means = self.get_means(dropped, hist_ims)
        num_frames, stack_size, height, width = downsampled_ims.shape
        downsampled_ims1 = np.reshape(downsampled_ims, (num_frames, stack_size, height * width))
        opti_inds = []
        ind = self.start_index
        opti_inds.append(ind)
        for j in range(1, 100):
            current_frame = downsampled_ims1[j]
            current_weighted_ims = []

            # current_map = current_map-0.10392
            for i in range(40):
                new_map = np.where(current_frame[i] > self.high_threshold, 0, 1)
                map_sum = np.sum(new_map)
                new_map = new_map * 18816 / map_sum
                current_weighted_ims.append(np.multiply(current_frame[i], new_map))
            current_weighted_ims = np.array(current_weighted_ims)

            the_means = np.mean(current_weighted_ims, axis=1)

            ind = 0
            min_residual = abs(the_means[0] - self.target_intensity)
            for i in range(1, 40):
                if abs(the_means[i] - self.target_intensity) < min_residual:
                    ind = i
                    min_residual = abs(the_means[i] - self.target_intensity)

            opti_inds.append(ind)

        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        hists_before_ds_outlier = np.zeros((100, 40, 101))

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier


    def pipeline_with_salient_map(self,salient_map):
            downsampled_ims = self.downsample_blending_rgb_channels()
            num_frames, stack_size, height, width = downsampled_ims.shape
            downsampled_ims1 = np.reshape(downsampled_ims,(num_frames,stack_size,height*width))
            opti_inds=[]
            ind = self.start_index
            print("INd", type(ind))
            opti_inds.append(ind)
            for j in range(1,100):
                current_frame = downsampled_ims1[j]
                current_map = np.reshape(salient_map[j-1][ind],(112*168))
                #current_map = current_map/np.sum(current_map)
                #current_weighted_ims = np.multiply(current_frame,current_map[None,:])
                current_weighted_ims = []

                #current_map = current_map-0.10392
                for i in range(40):
                    #
                    if j > 1:
                        pre_maps = np.empty((112,168,1))
                        for k in range(1):
                            pre_maps[:,:,k] = salient_map[j-k-1][opti_inds[j-k-1]]

                        #saliency = np.mean(pre_maps,axis=1).reshape(112*168)
                        saliency = salient_map[j-1][opti_inds[j-1]].reshape(112*168)
                    else:
                        saliency = np.array(current_map)

                    mask = np.where(saliency < 0.1, 0, 1)
                    combined = np.where(current_frame[i] > self.high_threshold, 0, mask)
                    total_number = len(saliency)
                    number_nonzeros = np.count_nonzero(combined)
                    salient_weight = 10/(total_number + number_nonzeros*9)
                    None_salient_weight = 1/(total_number + number_nonzeros*9)
                    new_map = np.where(combined == 0, None_salient_weight,salient_weight)
                    new_map = np.where(current_frame[i] > self.high_threshold, 0, new_map)
                    map_sum = np.sum(new_map)
                    new_map = new_map*18816/map_sum
                    current_weighted_ims.append(np.multiply(current_frame[i], new_map))
                current_weighted_ims = np.array(current_weighted_ims)



                the_means = np.mean(current_weighted_ims, axis=1)

                ind = 0
                min_residual = abs(the_means[0] - self.target_intensity)
                for i in range(1, 40):
                    if abs(the_means[i] - self.target_intensity) < min_residual:
                        ind = i
                        min_residual = abs(the_means[i] - self.target_intensity)

                opti_inds.append(ind)
            opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)


            weighted_means = np.zeros((100,40))
            hists = np.zeros((100,40,101))
            hists_before_ds_outlier = np.zeros((100,40,101))

            return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier


    def pipeline_local_without_grids_moving_object(self):
        downsampled_ims = self.downsample_blending_rgb_channels()
        local_area, local_area_before_outlier, global_area, global_area_before_outlier = self.get_flatten_weighted_imgs_local_wo_grids_moving_object_v2(
            downsampled_ims)

        # flatten_weighted_ims_before_ds_outlier = self.get_flatten_weighted_imgs(weights_before_ds_outlier, grided_ims)
        local_hists, local_dropped = self.get_hists(local_area)
        local_hists_before_ds_outlier, local_dropped_before_ds_outlier = self.get_hists(local_area_before_outlier)
        local_weighted_means = self.get_means(local_dropped, local_area)

        weighted_means = local_weighted_means
        opti_inds = self.get_optimal_img_index(weighted_means)
        opti_inds[0]=self.start_index

        hists_before_ds_outlier = np.zeros((100, 40, 101))
        hists = np.zeros((100, 40, 101))
        # opti_inds_adjusted = self.adjusted_opti_inds(opti_inds)
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier

