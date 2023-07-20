from exposure_class import Exposure


class Entropy(Exposure):
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
                start_index=20,):
        # self.gender = gender
        # Prototype initialization 3.x:
        super().__init__(
                input_images,
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
                start_index=start_index,)