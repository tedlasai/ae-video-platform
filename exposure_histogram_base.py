from exposure_general import Exposure

import numpy as np


class HistogramBase(Exposure):
    def __init__(self,
                 raw_images,
                 target_intensity=0.13,
                 high_threshold=1,
                 low_threshold=0,
                 start_index=20, ):
        super().__init__(
            raw_images,
            start_index=start_index,
        )
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.target_intensity = target_intensity

    def get_optimal_img_index(self, weighted_means):
        abs_weighted_errs_between_means_target = np.abs(weighted_means - self.target_intensity)
        return np.argmin(abs_weighted_errs_between_means_target)

    def produce_map(self, im):
        return np.ones(im.shape), im.shape[0] * im.shape[1]

    def get_means(self, good_pixel_ims, num_good_pixels):
        n_ev, h, w = good_pixel_ims.shape
        good_pixel_ims = np.reshape(good_pixel_ims, (n_ev, h * w))
        return np.sum(good_pixel_ims, axis=1) / np.array(num_good_pixels)
