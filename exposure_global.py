import numpy as np

from exposure_histogram_base import HistogramBase


class ExposureGlobal(HistogramBase):
    def __init__(self,
                 raw_images,
                 target_intensity=0.18,
                 high_threshold=1,
                 low_threshold=0,
                 start_index=20, ):

        super().__init__(
            raw_images,
            target_intensity=target_intensity,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            start_index=start_index,
        )

    def produce_map(self, im):
        new_map_step = np.where(im > self.high_threshold, 0, 1)
        new_map = np.where(im < self.low_threshold, 0, new_map_step)
        num_good_pixels = np.count_nonzero(new_map)
        return new_map, num_good_pixels

    def pipeline(self):
        downsampled_ims = self.raw_imgs
        weighted_means = []
        opti_inds = []
        ind = self.start_index
        hists = []
        for j in range(self.num_frame):
            current_frame = downsampled_ims[j]
            current_weighted_ims = []
            current_num_good_pixels = []
            curr_hists = []
            # current_map = current_map-0.10392
            for i in range(self.num_ims_per_frame):
                new_map, num_good_pixels = self.produce_map(current_frame[i])
                current_num_good_pixels.append(num_good_pixels)
                current_weighted_ims.append(np.multiply(current_frame[i], new_map))
                current_weighted_im_for_hist = np.where(new_map == 0, -0.01, current_frame[i])
                curr_hists.append(current_weighted_im_for_hist.flatten())
            frame_hists, num_dropped_pixels = self.get_hists_frame(np.array(curr_hists))
            hists.append(frame_hists)
            current_weighted_ims = np.array(current_weighted_ims)
            the_means = self.get_means(current_weighted_ims, current_num_good_pixels)
            weighted_means.append(the_means)
            ind = self.get_optimal_img_index(the_means)
            opti_inds.append(ind)
        opti_inds[0] = ind
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists
