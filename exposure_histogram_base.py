from exposure_general import Exposure


class HistogramBase(Exposure):
    def __init__(self,
                 raw_images,
                 srgb_imgs,
                 downsample_rate=1 / 64,
                 r_percent=0.25,
                 g_percent=0.5,
                 high_threshold=1,
                 low_threshold=0,
                 high_rate=0.2,
                 low_rate=0.2,
                 num_hist_bins=100,
                 stepsize=3,
                 number_of_previous_frames=5,
                 start_index=20, ):
        # self.gender = gender
        # Prototype initialization 3.x:
        self.srgb_images = srgb_imgs
        super().__init__(
            raw_images,
            downsample_rate=downsample_rate,
            r_percent=r_percent,
            g_percent=g_percent,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            high_rate=high_rate,
            low_rate=low_rate,
            num_hist_bins=num_hist_bins,
            stepsize=stepsize,
            number_of_previous_frames=number_of_previous_frames,
            start_index=start_index, )