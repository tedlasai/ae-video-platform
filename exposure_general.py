from update_visulization import hist_laxis, get_hists

import numpy as np
import constants
import collections


class Exposure:
    def __init__(self,
                 raw_imgs,  # blended raw images 112_168
                 start_index=20, ):
        self.absolute_bit = 2 ** 8  # max bit number of the raw image
        self.raw_imgs = raw_imgs
        self.num_frame = self.raw_imgs.shape[0]
        self.num_ims_per_frame = self.raw_imgs.shape[1]
        self.h = self.raw_imgs.shape[2]  # hight of a downsampled image
        self.w = self.raw_imgs.shape[3]
        self.SCALE_LABELS = constants.SCALES
        self.indexes_out_of_40 = constants.indexes_out_of_40
        self.NEW_SCALES = constants.NEW_SCALES
        self.start_index = start_index

    def adjusted_opti_inds_v2_by_average_of_previous_n_frames(self, opti_inds):
        length = len(opti_inds)
        opti_inds_new = np.array(opti_inds)
        last_visited_indices = collections.deque([opti_inds[0], opti_inds[0], opti_inds[0]], maxlen=3)

        if length > 2:
            i = 1
            while i < length:
                sum_ = 0
                for index in last_visited_indices:
                    sum_ += index
                average_of_previous_n_frames = sum_ / 3
                if abs(average_of_previous_n_frames - opti_inds_new[i]) > 1:
                    next_visit_index = (last_visited_indices[-2] + 2 * last_visited_indices[-1] + 3 * opti_inds_new[i]) / 6
                    opti_inds_new[i] = next_visit_index
                else:
                    opti_inds_new[i] = opti_inds_new[i - 1]
                current_index = opti_inds_new[i]
                last_visited_indices.append(current_index)
                i += 1
        print("OPT NEW", np.round(opti_inds_new).astype(int))
        return opti_inds_new

    def get_hists_frame(self, flatten_weighted_ims):
        scene_hists_include_drooped_counts = hist_laxis(flatten_weighted_ims, constants.num_hist_bins + 2, (
            -0.01, 1.01))  # 2 extra bin is used to count the number of -0.01 and 1
        num_dropped_pixels = scene_hists_include_drooped_counts[:, 0]
        scene_hists = scene_hists_include_drooped_counts[:, 1:]
        return scene_hists, num_dropped_pixels
