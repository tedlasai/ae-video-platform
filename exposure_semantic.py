from exposure_histogram_base import HistogramBase
from update_visulization import get_hists

import numpy as np


class ExposureSemantic(HistogramBase):
    def __init__(self,
                 raw_images,
                 list_local,
                 target_intensity=0.18,
                 high_threshold=1,
                 low_threshold=0,
                 start_index=20, ):
        self.local_indices = list_local

        super().__init__(
            raw_images,
            target_intensity=target_intensity,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            start_index=start_index,
        )

    def get_optimal_img_index(self, weighted_means):
        abs_weighted_errs_between_means_target = np.abs(weighted_means - self.target_intensity)
        return np.argmin(abs_weighted_errs_between_means_target, axis=1)

    def pipeline(self):
        downsampled_ims = self.raw_imgs
        local_area = self.get_flatten_weighted_imgs_local_wo_grids_moving_object_v2(
            downsampled_ims)
        local_hists, local_dropped = get_hists(local_area)
        local_weighted_means = self.get_means(local_dropped, local_area)

        weighted_means = local_weighted_means
        opti_inds = self.get_optimal_img_index(weighted_means)
        opti_inds[0] = self.start_index * 1.0
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, local_hists,  # hists_before_ds_outlier

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
        local_area = local_area.reshape((self.num_frame, self.num_ims_per_frame, self.h * self.w))

        local_area[local_area < self.low_threshold] = -0.01
        local_area[local_area > self.high_threshold] = -0.01
        return local_area

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
