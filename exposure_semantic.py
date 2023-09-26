import numpy as np

from exposure_histogram_base import HistogramBase


class ExposureSemantic(HistogramBase):
    def __init__(self,
                 raw_images,

                 list_local,
                 # downsample_rate=1 / 64,
                 target_intensity=0.18,
                 high_threshold=1,
                 low_threshold=0,
                 # high_rate=0.2,
                 # low_rate=0.2,
                 num_hist_bins=100,
                 # stepsize=3,
                 # number_of_previous_frames=5,
                 start_index=20, ):
        # self.gender = gender
        # Prototype initialization 3.x:
        self.local_indices = list_local

        super().__init__(
            raw_images,
            # downsample_rate=downsample_rate,


            target_intensity=target_intensity,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            # high_rate=high_rate,
            # low_rate=low_rate,
            num_hist_bins=num_hist_bins,

            start_index=start_index,

        )

        # def imput_imgs_processing(self):
        #     raw_bayer = np.load(self.srgb_images)
        #     # raw_bayer = raw_bayer[:, :, ::8, ::8, :]  # downsize 1/8 * 1/8 = 1/64, to be modified based on downsample size!!
        #     current_rgb_img = raw_bayer / (self.absolute_bit - 1)
        #     current_rgb_img[:, :, :, :, 0] = current_rgb_img[:, :, :, :, 0] * self.r_percent  # 0.2126 by default
        #     current_rgb_img[:, :, :, :, 1] = current_rgb_img[:, :, :, :, 1] * self.g_percent  # 0.7152 by default
        #     current_rgb_img[:, :, :, :, 2] = current_rgb_img[:, :, :, :, 2] * self.b_percent  # 0.0722 by default
        #     rgb_blended_ims = np.sum(current_rgb_img, axis=4)
        #     return rgb_blended_ims


    def get_optimal_img_index(self, weighted_means):
        abs_weighted_errs_between_means_target = np.abs(weighted_means - self.target_intensity)
        return np.argmin(abs_weighted_errs_between_means_target, axis=1)

    # def produce_map(self, im):
    #     new_map_step = np.where(im > self.high_threshold, 0, 1)
    #     new_map = np.where(im < self.low_threshold, 0, new_map_step)
    #     # map_sum = np.sum(new_map)
    #     # new_map = new_map #* 18816 / map_sum
    #     # new_map = np.where(current_frame[i] > self.high_threshold, 0, new_map)
    #     # new_map = np.where(current_frame[i] < self.low_threshold, 0, new_map)
    #     num_good_pixels = np.count_nonzero(new_map)
    #     return new_map, num_good_pixels

    # def get_means(self):
    #     return np.zeros(100)

    # def pipeline(self):
    #     super().pipeline()

    def pipeline(self):
        downsampled_ims = self.raw_imgs

        local_area = self.get_flatten_weighted_imgs_local_wo_grids_moving_object_v2(
            downsampled_ims)

        # flatten_weighted_ims_before_ds_outlier = self.get_flatten_weighted_imgs(weights_before_ds_outlier, grided_ims)
        local_hists, local_dropped = self.get_hists(local_area)
        # local_hists_before_ds_outlier, local_dropped_before_ds_outlier = self.get_hists(local_area_before_outlier)
        local_weighted_means = self.get_means(local_dropped, local_area)

        weighted_means = local_weighted_means
        opti_inds = self.get_optimal_img_index(weighted_means)
        opti_inds[0] = self.start_index*1.0

        # hists_before_ds_outlier = np.zeros((100, 40, 101))
        # hists = np.zeros((100, 40, 101))
        # opti_inds_adjusted = self.adjusted_opti_inds(opti_inds)
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, local_hists, #hists_before_ds_outlier

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
        # if self.global_rate > 0:
        #     global_area = np.where(local_area == -0.01, ims, -0.01)
        # else:
        # global_area = np.ones((self.num_frame, self.num_ims_per_frame, self.h, self.w)) * (-0.01)
        local_area = local_area.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))
        # local_area_before_outlier = np.array(local_area)
        local_area[local_area < self.low_threshold] = -0.01
        local_area[local_area > self.high_threshold] = -0.01

        # global_area = global_area.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))
        # global_area_before_outlier = np.array(global_area)
        # global_area[global_area < self.low_threshold] = -0.01
        # global_area[global_area > self.high_threshold] = -0.01

        return local_area #, local_area_before_outlier, global_area, global_area_before_outlier


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
        # weighted_means = weighted_means_flatten.reshape(self.num_frame, self.num_ims_per_frame)

        num_good_pixels = flatten_weighted_ims >= 0
        num_good_pixels = np.sum(num_good_pixels, axis=2)
        num_good_pixels[
            num_good_pixels == 0] = -1  # this allows for division when values are 0(usally really bright stuff)
        good_pixel_ims = flatten_weighted_ims
        good_pixel_ims[good_pixel_ims < 0] = 0  # make all the negative 0.01s to 0

        weighted_means = np.sum(good_pixel_ims, axis=2) / num_good_pixels

        return weighted_means
