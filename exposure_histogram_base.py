import numpy as np

from exposure_general import Exposure


class HistogramBase(Exposure):
    def __init__(self,
                 raw_images,

                 # downsample_rate=1 / 64,
                 target_intensity=0.13,
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

        super().__init__(
            raw_images,
            start_index=start_index,

        )
        self.low_threshold = low_threshold  # low outlier threshold
        self.high_threshold = high_threshold
        # self.high_rate = high_rate  # down sample rate of the areas over high outlier threshold
        # self.low_rate = low_rate
        self.target_intensity = target_intensity

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
        return np.argmin(abs_weighted_errs_between_means_target)

    def produce_map(self, im):
        return np.ones(im.shape),im.shape[0]*im.shape[1]

    def get_means(self,good_pixel_ims,num_good_pixels):
        n_ev,h,w = good_pixel_ims.shape
        good_pixel_ims = np.reshape(good_pixel_ims,(n_ev,h*w))

        return np.sum(good_pixel_ims, axis=1) / np.array(num_good_pixels)




