import numpy as np

from exposure_histogram_base import HistogramBase


class ExposureSaliency(HistogramBase):
    def __init__(self,
                 raw_images,
                 srgb_imgs,
                 salient_map,
                 # downsample_rate=1 / 64,
                 target_intensity=0.18,
                 high_threshold=1,
                 low_threshold=0,
                 high_rate=0.2,
                 low_rate=0.2,
                 num_hist_bins=100,
                 # stepsize=3,
                 # number_of_previous_frames=5,
                 start_index=20, ):
        # self.gender = gender
        # Prototype initialization 3.x:
        self.salient_map = salient_map

        super().__init__(
            raw_images,
            # downsample_rate=downsample_rate,

            srgb_imgs,
            target_intensity=target_intensity,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            high_rate=high_rate,
            low_rate=low_rate,
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


    def produce_map(self, im):
        new_map_step = np.where(im > self.high_threshold, 0, 1)
        new_map = np.where(im < self.low_threshold, 0, new_map_step)
        # map_sum = np.sum(new_map)
        # new_map = new_map #* 18816 / map_sum
        # new_map = np.where(current_frame[i] > self.high_threshold, 0, new_map)
        # new_map = np.where(current_frame[i] < self.low_threshold, 0, new_map)
        num_good_pixels = np.count_nonzero(new_map)
        return new_map,num_good_pixels