import numpy as np

from exposure_general import Exposure


class HistogramBase(Exposure):
    def __init__(self,
                 raw_images,

                 # downsample_rate=1 / 64,
                 target_intensity=0.13,
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

        super().__init__(
            raw_images,
            # downsample_rate=downsample_rate,


            num_hist_bins=num_hist_bins,

            start_index=start_index,

        )
        self.low_threshold = low_threshold  # low outlier threshold
        self.high_threshold = high_threshold
        self.high_rate = high_rate  # down sample rate of the areas over high outlier threshold
        self.low_rate = low_rate
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

    def pipeline(self):
        # downsampled_ims = self.downsample_blending_rgb_channels()
        # # generate histograms
        # hist_ims = np.array(downsampled_ims)
        # hist_ims[hist_ims > self.high_threshold] = -0.01
        # hist_ims = np.reshape(hist_ims, (self.num_frame, self.num_ims_per_frame, self.h * self.w))
        # hists, dropped = self.get_hists(hist_ims)
        # weighted_means = self.get_means(dropped, hist_ims)
        downsampled_ims = self.raw_imgs
        num_frames, stack_size, height, width = downsampled_ims.shape
        # downsampled_ims1 = np.reshape(downsampled_ims, (num_frames, stack_size, height * width))
        weighted_means = []
        opti_inds = []
        ind = self.start_index
        opti_inds.append(ind)
        for j in range(num_frames):
            current_frame = downsampled_ims[j]
            current_weighted_ims = []
            current_num_good_pixels = []

            # current_map = current_map-0.10392
            for i in range(stack_size):
                # new_map_step = np.where(current_frame[i] > self.high_threshold, 0, 1)
                # new_map = np.where(current_frame[i] < self.low_threshold, 0, new_map_step)
                # map_sum = np.sum(new_map)
                # new_map = new_map #* 18816 / map_sum
                # new_map = np.where(current_frame[i] > self.high_threshold, 0, new_map)
                # new_map = np.where(current_frame[i] < self.low_threshold, 0, new_map)
                new_map,num_good_pixels = self.produce_map(current_frame[i])
                current_num_good_pixels.append(num_good_pixels)
                current_weighted_ims.append(np.multiply(current_frame[i], new_map))
            current_weighted_ims = np.array(current_weighted_ims)

            # num_good_pixels = np.logical_not(np.logical_or(downsampled_ims1[j] > self.high_threshold,
            #                                                downsampled_ims1[j] < self.low_threshold))
            # num_good_pixels = np.logical_and(downsampled_ims1[j] <= self.high_threshold,
            #                                  downsampled_ims1[j] >= self.low_threshold)
            # num_good_pixels = np.sum(num_good_pixels, axis=1)
            # num_good_pixels[
            #     num_good_pixels == 0] = -1  # this allows for division when values are 0(usally really bright stuff)
            # good_pixel_ims = current_weighted_ims
            # good_pixel_ims[good_pixel_ims < 0] = 0  # make all the negative 0.01s to 0
            # num_good_pixels =
            # the_means = np.sum(good_pixel_ims, axis=1) / num_good_pixels
            the_means = self.get_means(current_weighted_ims,current_num_good_pixels)
            weighted_means.append(the_means)
            ind = self.get_optimal_img_index(the_means)
            opti_inds.append(ind)
        # min_residual = abs(the_means[0] - self.target_intensity)
        # for i in range(1, 40):
        #     if abs(the_means[i] - self.target_intensity) < min_residual:
        #         ind = i
        #         min_residual = abs(the_means[i] - self.target_intensity)



        opti_inds_adjusted_previous_n_frames = self.adjusted_opti_inds_v2_by_average_of_previous_n_frames(opti_inds)
        hists_before_ds_outlier = np.zeros((100, 40, 101))
        # print("get global outputs")
        # x = int(round(
        #             ind = 0

        #
        # hist_ims = downsampled_ims1[59][x]
        # # hists, dropped = self.get_hists(hist_ims)
        # hist = self.hist_laxis(hist_ims, 100, (0, 1.0001))
        # print("s16_59_22_HIST")
        # print(hist)
        # print(np.sum(hist))
        # print("end global")
        # import pandas as pd
        # import openpyxl
        # x = np.rint(opti_inds_adjusted_previous_n_frames)
        # print(x)
        # df = pd.DataFrame(x)
        #
        # # saving the dataframe
        # df.to_csv('global_output.csv')

        return opti_inds_adjusted_previous_n_frames, opti_inds, weighted_means, hists_before_ds_outlier, hists_before_ds_outlier


