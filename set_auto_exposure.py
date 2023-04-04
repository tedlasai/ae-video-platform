import os
import button_functions
import exposure_class
import numpy as np

def setAutoExposure(self, dummy=False):
    self.current_auto_exposure = self.defAutoExposure.get()
    self.scene_index = self.scene.index(self.defScene.get())
    input_ims = 'Image_Arrays_exposure_new/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
    #self.check_num_grids()
    self.exposureParams = {"downsample_rate": 1 / 25, 'r_percent': 0.25, 'g_percent': 0.5,
                           'col_num_grids': self.col_num_grids, 'row_num_grids': self.row_num_grids,
                           'low_threshold': self.low_threshold.get(), 'start_index': float(self.start_index.get()),
                           'high_threshold': self.high_threshold.get(), 'high_rate': float(self.high_rate.get()),
                           'stepsize': self.stepsize_limit.get(),
                           "number_of_previous_frames": self.number_of_previous_frames.get(),
                           "global_rate": self.local_interested_global_area_percentage.get(),
                           "target_intensity": self.target_intensity.get()}
    if (self.current_auto_exposure == "Global"):
        button_functions.clear_rects(self)
        exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                            target_intensity=self.exposureParams['target_intensity'],
                                            r_percent=self.exposureParams['r_percent'],
                                            g_percent=self.exposureParams['g_percent'],
                                            col_num_grids=self.exposureParams['col_num_grids'],
                                            row_num_grids=self.exposureParams['row_num_grids'],
                                            low_threshold=self.exposureParams['low_threshold'],
                                            start_index=self.exposureParams['start_index'],
                                            high_threshold=self.exposureParams['high_threshold'],
                                            high_rate=self.exposureParams['high_rate'],
                                            stepsize=self.exposureParams['stepsize'],
                                            number_of_previous_frames=self.exposureParams[
                                                'number_of_previous_frames'])
        # exposures = exposure_class.Exposure(params = self.exposureParams)

        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline()

    elif (self.current_auto_exposure == "Saliency_map"):
        button_functions.clear_rects(self)
        #name1 = self.scene[self.scene_index] + "_salient_maps_rbd.npy"
        map_name = self.scene[self.scene_index] + "_salient_maps_mbd.npy"

        # print(name)
        # salient_map_rbd = np.load("saliency_maps/"+name1)
        salient_map_mbd = np.load("saliency_maps/" + map_name)
        # salient_map = (salient_map_mbd +salient_map_rbd)/2
        salient_map = salient_map_mbd
        # print(self.scene[self.scene_index] + "_salient_maps_rbd.npy")
        # salient_map = np.load("Scene22_salient_maps_rbd.npy")
        exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                            target_intensity=self.exposureParams['target_intensity'],
                                            r_percent=self.exposureParams['r_percent'],
                                            g_percent=self.exposureParams['g_percent'],
                                            col_num_grids=self.exposureParams['col_num_grids'],
                                            row_num_grids=self.exposureParams['row_num_grids'],
                                            low_threshold=self.exposureParams['low_threshold'],
                                            start_index=self.exposureParams['start_index'],
                                            high_threshold=self.exposureParams['high_threshold'],
                                            high_rate=self.exposureParams['high_rate'],
                                            stepsize=self.exposureParams['stepsize'],
                                            number_of_previous_frames=self.exposureParams[
                                                'number_of_previous_frames'])
        # exposures = exposure_class.Exposure(params = self.exposureParams)

        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_with_salient_map(
            salient_map)

    elif (self.current_auto_exposure == "Entropy"):
        button_functions.clear_rects(self)
        srgb_ims = 'Image_Arrays_from_dng/Scene' + str(self.scene_index + 1) + '_show_dng_imgs.npy'
        exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                            target_intensity=self.exposureParams['target_intensity'],
                                            r_percent=self.exposureParams['r_percent'],
                                            g_percent=self.exposureParams['g_percent'],
                                            low_threshold=0,
                                            start_index=self.exposureParams['start_index'],
                                            high_threshold=1.0,
                                            high_rate=0,
                                            stepsize=self.exposureParams['stepsize'],
                                            number_of_previous_frames=self.exposureParams[
                                                'number_of_previous_frames'])
        # exposures = exposure_class.Exposure(params = self.exposureParams)
        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.entropy_pipeline(
            srgb_ims)

    elif (self.current_auto_exposure == "Max Gradient srgb"):
        button_functions.clear_rects(self)
        input_ims = 'Image_Arrays_from_dng/Scene' + str(self.scene_index + 1) + '_show_dng_imgs.npy'
        exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                            target_intensity=self.exposureParams['target_intensity'],
                                            r_percent=self.exposureParams['r_percent'],
                                            g_percent=self.exposureParams['g_percent'],
                                            low_threshold=0,
                                            start_index=self.exposureParams['start_index'],
                                            high_threshold=1.0,
                                            high_rate=0,
                                            stepsize=self.exposureParams['stepsize'],
                                            number_of_previous_frames=self.exposureParams[
                                                'number_of_previous_frames'])
        # exposures = exposure_class.Exposure(params = self.exposureParams)
        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.gradient_srgb_exposure_pipeline()

    elif (self.current_auto_exposure == "HDR Histogram Method"):
        button_functions.clear_rects(self)
        exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                            target_intensity=self.exposureParams['target_intensity'],
                                            r_percent=self.exposureParams['r_percent'],
                                            g_percent=self.exposureParams['g_percent'],
                                            low_threshold=0,
                                            start_index=self.exposureParams['start_index'],
                                            high_threshold=1.0,
                                            high_rate=0,
                                            stepsize=self.exposureParams['stepsize'],
                                            number_of_previous_frames=self.exposureParams[
                                                'number_of_previous_frames'])
        # exposures = exposure_class.Exposure(params = self.exposureParams)
        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.hdr_max_area_pipeline()

    elif (self.current_auto_exposure == "Local on moving objects"):
        button_functions.clear_rects_local(self)
        list_local = self.list_local_without_grids_moving_objects()

        import pickle

        name = f"local_pickle/Scene{self.scene_index + 1}.pkl"
        print("LIST LOCAL", list_local)
        with open(name, 'wb') as handle:
            pickle.dump({'boxes': list_local}, handle, protocol=pickle.HIGHEST_PROTOCOL)

        exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                            target_intensity=self.exposureParams['target_intensity'],
                                            r_percent=self.exposureParams['r_percent'],
                                            g_percent=self.exposureParams['g_percent'],
                                            col_num_grids=self.exposureParams['col_num_grids'],
                                            row_num_grids=self.exposureParams['row_num_grids'],
                                            low_threshold=self.exposureParams['low_threshold'],
                                            start_index=self.exposureParams['start_index'],
                                            high_threshold=self.exposureParams['high_threshold'],
                                            high_rate=self.exposureParams['high_rate'], local_indices=list_local,
                                            stepsize=self.exposureParams['stepsize'],
                                            number_of_previous_frames=self.exposureParams[
                                                'number_of_previous_frames'],
                                            global_rate=self.exposureParams['global_rate']
                                            )
        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_local_without_grids_moving_object()
        # print("list_local:")
        # print(list_local)
    # print("CURRENT AUTO EXPOSURE", self.current_auto_exposure)
    # print("adjusted_by_previous_n_frames")
    # print(self.eV)
    # print("original_output")
    # print(self.eV_original)


def setValues(self, dummy=False):
    button_functions.runVideo(self)
    # self.play = True
    # self.playVideo()
    # time.sleep(1)
    # print(scene_name)
    # self.check_num_grids()
    if self.scene[self.scene_index] != self.defScene.get():
        button_functions.clear_rects(self)
        self.scene_index = self.scene.index(self.defScene.get())
        # input_ims = 'Image_Arrays_exposure/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        setAutoExposure(self)

        # exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"], r_percent=self.exposureParams['r_percent'], g_percent=self.exposureParams['g_percent'],
        #                                     col_num_grids=self.exposureParams['col_num_grids'], row_num_grids=self.exposureParams['row_num_grids'], low_threshold=self.exposureParams['low_threshold'], start_index=self.exposureParams['start_index'],
        #                                     high_threshold=self.exposureParams['high_threshold'], high_rate=self.exposureParams['high_rate'])
        # self.eV,weighted_means,hists,hists_before_ds_outlier = exposures.pipeline()

        # self.img_mean_list = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.defScene.get() + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2 ** self.bit_depth - 1)

        # self.img_mertens = np.load(
        #     os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[
        #         self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')

        self.img_raw = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                self.scene_index] + '_show_dng_imgs' + '.npy')
        self.img_all = self.img_raw
        button_functions.resetValues(self)