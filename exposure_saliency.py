from exposure_histogram_base import HistogramBase

import numpy as np


class ExposureSaliency(HistogramBase):
    def __init__(self,
                 raw_images,
                 salient_map,
                 target_intensity=0.18,
                 high_threshold=1,
                 low_threshold=0,
                 start_index=20,
                 smoothness_number=3):
        self.salient_map = salient_map
        self.salient_pix_ratio = 14
        self.non_salient_pix_ration = 1

        super().__init__(
            raw_images,
            target_intensity=target_intensity,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            start_index=start_index,
            smoothness_number=smoothness_number,
        )

    def pipeline(self):
        downsampled_ims = self.raw_imgs
        total_n_pixs = self.h * self.w
        downsampled_ims1 = np.reshape(downsampled_ims,(self.num_frame,self.num_ims_per_frame,total_n_pixs))
        opti_inds=[]
        ind = int(self.start_index)
        opti_inds.append(ind)
        first_frame_flatten_ims = downsampled_ims1[0]
        first_frame_means = np.mean(first_frame_flatten_ims,axis=1)
        first_frame_hists,_ = self.get_hists_frame(first_frame_flatten_ims)
        means = [first_frame_means]
        hists = [first_frame_hists]
        for j in range(1,self.num_frame):
            current_frame = downsampled_ims1[j]
            current_map = np.reshape(self.salient_map[j-1][ind], total_n_pixs)
            current_weighted_ims = []
            for i in range(self.num_ims_per_frame):
                if j > 1:
                    pre_maps = np.empty((self.h,self.w,2))
                    for k in range(2):
                        pre_maps[:,:,k] = self.salient_map[j-k-1][opti_inds[j-k-1]]
                    saliency = self.salient_map[j-1][opti_inds[j-1]].reshape(total_n_pixs)
                else:
                    saliency = np.array(current_map)
                mask = np.where(saliency < 0.1, 0, 1)
                combined = np.where(current_frame[i] > self.high_threshold, 0, mask)
                combined = np.where(current_frame[i] < self.low_threshold, 0, combined)
                number_nonzeros = np.count_nonzero(combined) #number of salient pixels between the thresholds
                total_n_pixs_weighted = total_n_pixs + number_nonzeros*(self.salient_pix_ratio-1)
                salient_weight = self.salient_pix_ratio/total_n_pixs_weighted
                None_salient_weight = self.non_salient_pix_ration/total_n_pixs_weighted
                new_map = np.where(combined == 0, None_salient_weight,salient_weight) #build a map with the salient weights
                map_sum = np.sum(new_map)
                new_map = (new_map/map_sum)*total_n_pixs
                current_weighted_ims.append(np.multiply(current_frame[i], new_map))
            current_weighted_ims = np.array(current_weighted_ims)
            frame_hists, _ = self.get_hists_frame(current_weighted_ims)
            frame_means = np.mean(current_weighted_ims, axis=1)
            ind = 0
            min_residual = abs(frame_means[0] - self.target_intensity)
            for i in range(1, 40):
                if abs(frame_means[i] - self.target_intensity) < min_residual:
                    ind = i
                    min_residual = abs(frame_means[i] - self.target_intensity)
            opti_inds.append(ind)
            means.append(frame_means)
            hists.append(frame_hists)
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v3_by_average_of_n_frames(opti_inds, self.smoothness_number)
        return opti_inds_adjusted_previous_n_frames, opti_inds, means, hists

