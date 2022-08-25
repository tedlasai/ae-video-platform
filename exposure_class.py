import math
import numpy as np

class Exposure:
    def __init__(self, input_images, downsample_rate=1 / 64, r_percent=0, g_percent=1, col_num_grids=8, row_num_grids=8,
                 target_intensity=0.19, high_threshold=1, low_threshold=0, high_rate=0.2, low_rate=0.2,
                 local_indices=[], num_hist_bins=100, local_with_downsampled_outliers=False):
        self.absolute_bit = 2**8  # max bit number of the raw image
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
            return matrix[:, :, ::row_skip_step, 1::col_skip_step] / 2 / self.absolute_bit
        if channel == 'g2':
            return matrix[:, :, 1::row_skip_step, ::col_skip_step] / 2 / self.absolute_bit
        if channel == 'r':
            return matrix[:, :, ::row_skip_step, ::col_skip_step] / self.absolute_bit
        return matrix[:, :, 1::row_skip_step, 1::col_skip_step] / self.absolute_bit

    def downsample_blending_rgb_channels(self):
        if self.downsample_rate >= 0.75:
            self.downsample_rate = 1
        elif self.downsample_rate > 0.5:
            self.downsample_rate = 0.5
        sqrt_rate = math.sqrt(1 / self.downsample_rate)
        decimal_ = sqrt_rate - int(sqrt_rate)
        if decimal_ > 0.85 or decimal_ < 0.15:
            col_skip_step = round(sqrt_rate) * 2
            row_skip_step = col_skip_step * 2
        else:
            col_skip_step = math.ceil(sqrt_rate) * 2
            row_skip_step = math.floor(sqrt_rate) * 2

        raw_bayer = np.load(self.input_images)
        print("input shape")
        print(raw_bayer.shape)
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
        grid_means = np.reshape(temp_ims, (self.num_frame, self.num_ims_per_frame, self.row_num_grids, self.col_num_grids))
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

    def get_flatten_weighted_imgs_local_wo_grids(self, ims):
        flatten_weighted_ims = np.ones((self.num_frame, self.num_ims_per_frame, self.h , self.w)) * (-0.01)
        for (y_start,x_start,y_end,x_end) in self.local_indices:
            y_start = int(y_start * self.h)
            x_start = int(x_start * self.w)
            y_end = int(y_end * self.h)
            x_end = int(x_end * self.w)
            print("&&&&&")
            print(y_start)
            print(x_start)
            print(y_end)
            print(x_end)
            print("&&&&&&")

            flatten_weighted_ims[:,:,y_start:y_end+1,x_start:x_end+1] = ims[:,:,y_start:y_end+1,x_start:x_end+1]
        print(ims.shape)
        flatten_weighted_ims = flatten_weighted_ims.reshape((self.num_frame, self.num_ims_per_frame, self.h*self.w))
        flatten_weighted_ims_before_outlier = np.array(flatten_weighted_ims)
        flatten_weighted_ims[flatten_weighted_ims < self.low_threshold] = -0.01
        flatten_weighted_ims[flatten_weighted_ims > self.high_threshold] = -0.01
        print("high rate: "+ str(self.high_threshold))
        print("low rate: "+str(self.low_threshold))
        print(flatten_weighted_ims[0][0])
        print(flatten_weighted_ims_before_outlier[0][0])
        return flatten_weighted_ims,flatten_weighted_ims_before_outlier

    def get_hists(self, flatten_weighted_ims):
        scene_hists_include_drooped_counts = self.hist_laxis(flatten_weighted_ims, self.num_hist_bins + 1, (
        -0.01, 1))  # one extra bin is used to count the number of -0.01
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
        weighted_all_means = np.mean(flatten_weighted_ims, axis=2).reshape((self.num_frame, self.num_ims_per_frame, 1))
        num_dropped_pixels = num_dropped_pixels.reshape((self.num_frame, self.num_ims_per_frame, 1))
        concat_weighted_means_num_dropped_pixels = np.concatenate((weighted_all_means, num_dropped_pixels), axis=2)
        weighted_means_flatten = np.apply_along_axis(self.correct_one_mean, 2, concat_weighted_means_num_dropped_pixels)
        weighted_means = weighted_means_flatten.reshape(self.num_frame, self.num_ims_per_frame)
        return weighted_means

    def get_optimal_img_index(self, weighted_means):
        abs_weighted_errs_between_means_target = np.abs(weighted_means - self.target_intensity)
        return np.argmin(abs_weighted_errs_between_means_target, axis=1)

    def pipeline(self):
        downsampled_ims = self.downsample_blending_rgb_channels()
        grided_ims, grided_means = self.get_grided_ims(downsampled_ims)
        weights,weights_before_ds_outlier = self.get_grids_weight_matrix(grided_means)
        flatten_weighted_ims = self.get_flatten_weighted_imgs(weights, grided_ims)
        flatten_weighted_ims_before_ds_outlier = self.get_flatten_weighted_imgs(weights_before_ds_outlier, grided_ims)
        hists, dropped = self.get_hists(flatten_weighted_ims)
        hists_before_ds_outlier, dropped_before_ds_outlier = self.get_hists(flatten_weighted_ims_before_ds_outlier)
        weighted_means = self.get_means(dropped, flatten_weighted_ims)
        opti_inds = self.get_optimal_img_index(weighted_means)
        return opti_inds,weighted_means,hists,hists_before_ds_outlier

    def pipeline_local_without_grids(self):
        downsampled_ims = self.downsample_blending_rgb_channels()
        # grided_ims, grided_means = self.get_grided_ims(downsampled_ims)
        # weights,weights_before_ds_outlier = self.get_grids_weight_matrix(grided_means)
        flatten_weighted_ims,flatten_weighted_ims_before_outlier = self.get_flatten_weighted_imgs_local_wo_grids(downsampled_ims)

        #flatten_weighted_ims_before_ds_outlier = self.get_flatten_weighted_imgs(weights_before_ds_outlier, grided_ims)
        hists, dropped = self.get_hists(flatten_weighted_ims)
        hists_before_ds_outlier, dropped_before_ds_outlier = self.get_hists(flatten_weighted_ims_before_outlier)
        weighted_means = self.get_means(dropped, flatten_weighted_ims)
        opti_inds = self.get_optimal_img_index(weighted_means)
        return opti_inds,weighted_means,hists,hists_before_ds_outlier