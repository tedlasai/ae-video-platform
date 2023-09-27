from update_visulization import hist_laxis, get_hists

import numpy as np
import constants
import collections

class Exposure:
    def __init__(self,
                raw_imgs, # blended raw images 112_168
                start_index =20,):
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

    # helper function to add two 4d arrays those might have different shape in 3red and 4th dimrntions(trim thr larger one)

    # def initial_functions(self):
    #     #self.compute_downsample_size()
    #     pass

    def adjusted_opti_inds_v2_by_average_of_previous_n_frames(self, opti_inds):
        length = len(opti_inds)
        opti_inds_new = np.array(opti_inds)
        print("# previes frames")
        print(opti_inds)
        #current_index = opti_inds_new[0]
        # if length > 1:
        #     i = 1
        #     while i < length:
        #         start_index = max(0, i - self.number_of_previous_frames)
        #         # print("start_ind: "+str(start_index))
        #         average_of_previous_n_frames = np.mean(opti_inds_new[start_index:i])
        #         diff = average_of_previous_n_frames - opti_inds_new[i]
        #         if diff < -self.stepsize:
        #             opti_inds_new[i] = round(average_of_previous_n_frames + self.stepsize)
        #         if diff > self.stepsize:
        #             opti_inds_new[i] = round(average_of_previous_n_frames - self.stepsize)
        #         i += 1

        lastVisitedIndices = collections.deque([opti_inds[0],opti_inds[0],opti_inds[0]], maxlen=3)

        if length > 2:
            i = 1
            while i < length:
                # print("start_ind: "+str(start_index))
                sum = 0
                for index in lastVisitedIndices:
                    sum+=index
                average_of_previous_n_frames = sum/3
                #print(f"i{i}, average_of_previous_n_frames {average_of_previous_n_frames},  opti_inds_new[i] { opti_inds_new[i]}")
                if(abs(average_of_previous_n_frames - opti_inds_new[i] )> 1):
                   # print(lastVisitedIndices)
                    nextVisitIndex = (lastVisitedIndices[-2] + 2*lastVisitedIndices[-1] +3*opti_inds_new[i])/6
                    #current_index = opti_inds_new[i]
                    opti_inds_new[i] = nextVisitIndex
                else:
                    opti_inds_new[i] = opti_inds_new[i-1]
                current_index = opti_inds_new[i]
                lastVisitedIndices.append(current_index)



                #override change
                #if abs(opti_inds_new[i]-opti_inds_new[i-1]) < 3 and( opti_inds_new[i-1]-opti_inds_new[i-2]) < 2:
                #    opti_inds_new[i] = opti_inds_new[i-1]
                i += 1
        print("OPT NEW", opti_inds_new)
        return opti_inds_new

    # def get_hists(self, flatten_weighted_ims):
    #     scene_hists_include_drooped_counts = hist_laxis(flatten_weighted_ims, self.num_hist_bins + 2, (
    #         -0.01, 1.01))  # 2 extra bin is used to count the number of -0.01 and 1
    #     num_dropped_pixels = scene_hists_include_drooped_counts[:, :, 0]
    #     scene_hists = scene_hists_include_drooped_counts[:, :, 1:]
    #     return scene_hists, num_dropped_pixels

    def get_hists_frame(self, flatten_weighted_ims):
        scene_hists_include_drooped_counts = hist_laxis(flatten_weighted_ims, self.num_hist_bins + 2, (
            -0.01, 1.01))  # 2 extra bin is used to count the number of -0.01 and 1
        num_dropped_pixels = scene_hists_include_drooped_counts[:, 0]
        scene_hists = scene_hists_include_drooped_counts[:, 1:]
        return scene_hists, num_dropped_pixels

    # def hist_laxis(self, data, n_bins, range_limits):
    #     return hist_laxis(data, n_bins, range_limits)


