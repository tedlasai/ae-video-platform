import numpy as np

from exposure_histogram_base import HistogramBase


class ExposureSaliency(HistogramBase):
    def __init__(self,
                 raw_images,

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
        self.salient_pix_ratio = 14
        self.non_salient_pix_ration = 1

        super().__init__(
            raw_images,
            # downsample_rate=downsample_rate,


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

    def pipeline(self):
        print('here')
        downsampled_ims = self.raw_imgs
        total_n_pixs = self.h * self.w

        downsampled_ims1 = np.reshape(downsampled_ims,(self.num_frame,self.num_ims_per_frame,total_n_pixs))
        opti_inds=[]
        ind = int(self.start_index)
        opti_inds.append(ind)

        for j in range(1,100):
            current_frame = downsampled_ims1[j]
            current_map = np.reshape(self.salient_map[j-1][ind],(total_n_pixs))
            #current_map = current_map/np.sum(current_map)
            #current_weighted_ims = np.multiply(current_frame,current_map[None,:])
            current_weighted_ims = []

            #current_map = current_map-0.10392
            for i in range(40):
                #
                if j > 1:
                    pre_maps = np.empty((self.h,self.w,2))
                    for k in range(2):
                        pre_maps[:,:,k] = self.salient_map[j-k-1][opti_inds[j-k-1]]

                    # saliency = np.mean(pre_maps,axis=2).reshape(height * width)
                    saliency = self.salient_map[j-1][opti_inds[j-1]].reshape(total_n_pixs)
                else:
                    saliency = np.array(current_map)

                mask = np.where(saliency < 0.1, 0, 1)
                combined = np.where(current_frame[i] > self.high_threshold, 0, mask)
                combined = np.where(current_frame[i] < self.low_threshold, 0, combined)
                total_number = len(saliency)
                number_nonzeros = np.count_nonzero(combined) #number of salient pixels between the thresholds
                salient_weight = self.salient_pix_ratio/(total_number + number_nonzeros*13)
                None_salient_weight = self.non_salient_pix_ration/(total_number + number_nonzeros*13)
                new_map = np.where(combined == 0, None_salient_weight,salient_weight) #build a map with the salient weights
                #zero out stuff above and below threshold
                new_map = np.where(current_frame[i] > self.high_threshold, 0, new_map)
                new_map = np.where(current_frame[i] < self.low_threshold, 0, new_map)
                map_sum = np.sum(new_map)
                num_good_pixels = np.logical_not(np.logical_or(current_frame[i] > self.high_threshold,
                                                               current_frame[i] < self.low_threshold))
                num_good_pixels = np.sum(num_good_pixels)
                new_map_ = new_map*(total_number + number_nonzeros*13)
                new_map = (new_map/map_sum)*total_n_pixs #normalizes the map(this might be wrong)



                current_weighted_ims.append(np.multiply(current_frame[i], new_map))
            current_weighted_ims = np.array(current_weighted_ims)

            # num_good_pixels = np.logical_not(np.logical_or(downsampled_ims1[j] > self.high_threshold,
            #                                 downsampled_ims1[j] < self.low_threshold))
            # num_good_pixels = np.sum(num_good_pixels, axis=1)
            # num_good_pixels[
            #     num_good_pixels == 0] = -1  # this allows for division when values are 0(usally really bright stuff)
            # good_pixel_ims = current_weighted_ims
            # good_pixel_ims[good_pixel_ims < 0] = 0  # make all the negative 0.01s to 0
            #
            # the_means = np.sum(good_pixel_ims, axis=1)/num_good_pixels

            #I THINK ALL YOU NEED IS
            the_means = np.mean(current_weighted_ims, axis=1)

            #the_means = np.mean(current_weighted_ims, axis=1)

            ind = 0
            min_residual = abs(the_means[0] - self.target_intensity)
            for i in range(1, 40):
                if abs(the_means[i] - self.target_intensity) < min_residual:
                    ind = i
                    min_residual = abs(the_means[i] - self.target_intensity)

            opti_inds.append(ind)
        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)


        weighted_means = np.zeros((100,40))
        hists = np.zeros((100,40,101))
        hists_before_ds_outlier = np.zeros((100,40,101))

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists, hists_before_ds_outlier

