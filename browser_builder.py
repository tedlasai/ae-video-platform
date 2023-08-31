import constants
import tkinter as tk

import exposure_global
import set_auto_exposure
import pandas as pd
from RangeSlider.RangeSlider import RangeSliderH
from PIL import Image, ImageTk
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import glob
import platform
import regular
import high_res_auto_ex_video
import exposure_class
from test_pipline import local_interested_grids_generater
import pickle as pkl
import button_functions
import manual_semantic_functions


# root = tk.Tk()
# root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5
class Browser:

    def __init__(self, root):
        super().__init__()
        self.blended_raw_ims = None
        self.root = root
        self.folders = constants.folders
        self.widgetFont = constants.widgetFont
        self.widgetFontSize = constants.widgetFontSize
        self.scene = constants.scene
        self.frame_num = constants.frame_num
        self.stack_size = constants.stack_size
        self.SCALE_LABELS = constants.SCALE_LABELS
        self.SCALE_LABELS_NEW = constants.SCALE_LABELS_NEW
        self.NEW_SCALES = constants.NEW_SCALES
        self.eV = []
        self.auto_exposures = constants.auto_exposures
        self.current_auto_exposure = "None"
        self.scene_index = 18
        # self.mertensVideo = []
        # self.bit_depth = 8
        # self.downscale_ratio = 0.12
        # self.check = True
        self.temp_img_ind = 0
        self.making_a_serious_of_videos = 0
        self.red_ratio = 0.25
        self.green_ratio = 0.5
        self.blue_ratio = 0.25

        self.joinPathChar = "/"
        if (platform.system() == "Windows"):
            self.joinPathChar = "\\"

        self.imgSize = constants.imgSize
        self.widthToScale = self.imgSize[1]
        self.widPercent = (self.widthToScale / float(self.imgSize[1]))
        self.heightToScale = int(float(self.imgSize[0]) * float(self.widPercent))

        self.img_all = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                self.scene_index] + '_show_dng_imgs' + '.npy')

        self.raw_ims = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays_exposure_224_336') + self.joinPathChar + self.scene[
                self.scene_index] + '_ds_raw_imgs' + '.npy')

        self.img = deepcopy(self.img_all[0, 0])
        self.play = True
        self.video_speed = 50
        self.video_fps = 30

        # self.res_check = 0
        # self.hdr_mode_check = 0
        self.make_crop_video_flag = 0
        self.temp_coords_for_video_producing = []

        # Image Convas
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.img).resize((self.imgSize[1], self.imgSize[0])))

        self.canvas = tk.Canvas(root, cursor="cross", width=self.photo.width(), height=self.photo.height(),
                                borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=1, column=1, columnspan=2, rowspan=27, padx=0, pady=0, sticky=tk.NW)
        self.canvas_img = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.current_rects = []  # the rectangles drawn in canvas
        self.rectangles = []  # the coordinates of the rectangles
        self.current_rects_wo_grids = []
        self.rects_without_grids = []  # the coordinates of the rectangles @ local without grids
        self.rects_without_grids_moving_objests = {}  # key: frame number, value: list of rect ids
        self.moving_rectids = []
        self.the_moving_rect = None
        self.the_scrolling_rect = None
        # self.canvas.bind('<Button-1>', self.canvas_click)
        # self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        # self.canvas.bind('<B1-Motion>', self.on_move_press)
        # self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        # self.canvas.bind("<Button-3>", self.right_click)
        # self.canvas.bind("<MouseWheel>", self.zoomer)
        # self.canvas.bind("<Button-4>", self.zoomerP)
        # self.canvas.bind("<Button-5>", self.zoomerM)
        # some defaults
        self.col_num_grids = 8
        self.row_num_grids = 8
        self.rowGridSelect = 0
        self.colGridSelect = 0
        self.rect = None
        self.x = 0
        self.y = 0
        self.start_x = None
        self.start_y = None
        self.curX = 0
        self.curY = 0
        self.num_bins = constants.num_bins
        self.hists = []
        self.hists_before_ds_outlier = []
        self.fig_2 = None
        self.fig = None
        self.fig_4 = None
        self.local_consider_outliers_check = 0
        self.show_srgb_hist_check = 0
        self.srgb_mean = 0
        self.check = True
        # self.init_functions()
        # self.show_srgb_hist_check = self.show_srgb_hist_check_.get()

    def init_functions(self):
        self.scene_select()
        self.auto_exposure_select()
        self.blend_rgb()

    def blend_rgb(self):
        self.red_ratio = 0.25
        self.green_ratio = 0.5
        self.blue_ratio = 0.25
        print(type(self.raw_ims))
        print(self.raw_ims.shape)
        self.blended_raw_ims = (self.raw_ims[:, :, ::2, 1::2] + self.raw_ims[:, :, 1::2, ::2]) * self.green_ratio / 2 \
                               + self.raw_ims[:, :, ::2, ::2] * self.red_ratio + self.raw_ims[:, :, 1::2,
                                                                                 1::2] * self.blue_ratio

    def horizontal_slider(self, command_function):
        # Horizantal Slider
        self.horSlider = tk.Scale(self.root, activebackground='black', cursor='sb_h_double_arrow', from_=0,
                                  to=self.frame_num[0] - 1,
                                  label='Frame Number', font=(self.widgetFont, self.widgetFontSize),
                                  orient=tk.HORIZONTAL,
                                  length=self.widthToScale, command=command_function)
        self.horSlider.grid(row=27, column=1, columnspan=2, sticky=tk.SW)

    def vertical_slider(self, command_function):
        # Vertical Slider

        self.verSliderLabel = tk.Label(self.root, text='Exposure Time', font=(self.widgetFont, self.widgetFontSize))
        self.verSliderLabel.grid(row=0, column=0)

        min_ = 0
        max_ = self.stack_size[self.scene_index] - 1
        self.verSlider = tk.Scale(self.root, activebackground='black', cursor='sb_v_double_arrow', from_=min_, to=max_,
                                  font=(self.widgetFont, self.widgetFontSize),
                                  length=self.heightToScale,
                                  command=command_function)

        # print(self.verSlider.configure().keys())

        self.verSlider.grid(row=1, column=0, rowspan=25)

    def scene_select(self):
        # Select Scene List
        self.defScene = tk.StringVar(self.root)
        self.defScene.set(self.scene[self.scene_index])  # default value
        self.selSceneLabel = tk.Label(self.root, text='Select Scene:', font=(self.widgetFont, self.widgetFontSize))
        self.selSceneLabel.grid(row=0, column=3, sticky=tk.W)
        self.sceneList = tk.OptionMenu(self.root, self.defScene, *self.scene, command=self.setValues)
        self.sceneList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.sceneList.grid(row=1, column=3, sticky=tk.NE)

    def auto_exposure_select(self):
        # Select Scene List
        self.defAutoExposure = tk.StringVar(self.root)
        self.defAutoExposure.set(self.auto_exposures[0])  # default value
        self.selAutoExposureLabel = tk.Label(self.root, text='Select AutoExposure:',
                                             font=(self.widgetFont, self.widgetFontSize))
        self.selAutoExposureLabel.grid(row=0, column=4, sticky=tk.W)
        self.AutoExposureList = tk.OptionMenu(self.root, self.defAutoExposure, *self.auto_exposures,
                                              command=self.setAutoExposure)
        self.AutoExposureList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.AutoExposureList.grid(row=1, column=4, sticky=tk.NE)

    # def updateSlider(self, scale_value):
    #     pass
    #
    # def scale_labels(self, value):
    #     pass

    def buttons_builder(self, text, command_function, row, column, para=0):
        self.b = tk.Button(self.root, text=text,
                           fg='#ffffff',
                           bg='#999999',
                           activebackground='#454545',
                           relief=tk.RAISED,
                           width=16,
                           padx=10,
                           pady=5,
                           font=(constants.widgetFont, constants.widgetFontSize),
                           command=lambda: command_function(para))
        # self.b['command'] = lambda arg="live", kw="as the": command_function(arg, opt1=kw)
        self.b.grid(row=row, column=column, sticky=tk.E)

    def setAutoExposure(self, dummy=1):
        self.current_auto_exposure = self.defAutoExposure.get()
        self.scene_index = self.scene.index(self.defScene.get())
        #input_ims = 'Image_Arrays_exposure_new/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        input_ims = self.blended_raw_ims
        srgb_ims = self.img_all
        # self.check_num_grids()
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
            exposures = exposure_global.ExposureGlobal(input_ims, srgb_ims,
                                                target_intensity=self.exposureParams['target_intensity'],
                                                low_threshold=self.exposureParams['low_threshold'],
                                                start_index=self.exposureParams['start_index'],
                                                high_threshold=self.exposureParams['high_threshold'],
                                                high_rate=self.exposureParams['high_rate'],
                                              )
            # exposures = exposure_class.Exposure(params = self.exposureParams)

            self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline()

        elif (self.current_auto_exposure == "Saliency_map"):
            button_functions.clear_rects(self)
            # name1 = self.scene[self.scene_index] + "_salient_maps_rbd.npy"
            map_name = self.scene[self.scene_index] + "_salient_maps_mbd.npy"

            # print(name)
            # salient_map_rbd = np.load("saliency_maps/"+name1)
            salient_map_mbd = np.load("saliency_maps/" + map_name)
            # salient_map = (salient_map_mbd +salient_map_rbd)/2
            salient_map = salient_map_mbd
            # print(self.scene[self.scene_index] + "_salient_maps_rbd.npy")
            # salient_map = np.load("Scene22_salient_maps_rbd.npy")
            exposures = exposure_class.Exposure(input_ims, srgb_ims, salient_map,downsample_rate=self.exposureParams["downsample_rate"],
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

            self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_with_salient_map()

        elif (self.current_auto_exposure == "Entropy"):
            button_functions.clear_rects(self)
            srgb_ims = 'Image_Arrays_from_dng/Scene' + str(self.scene_index + 1) + '_show_dng_imgs.npy'
            exposures = exposure_class.Exposure(input_ims, srgb_ims, downsample_rate=self.exposureParams["downsample_rate"],
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
            exposures = exposure_class.Exposure(input_ims, srgb_ims, downsample_rate=self.exposureParams["downsample_rate"],
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
            exposures = exposure_class.Exposure(input_ims, srgb_ims, downsample_rate=self.exposureParams["downsample_rate"],
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

        elif (self.current_auto_exposure == "Semantic"):
            button_functions.clear_rects_local(self)
            list_local = manual_semantic_functions.list_local_without_grids_moving_objects(self)

            # import pickle
            #
            # name = f"local_pickle/Scene{self.scene_index + 1}.pkl"
            # print("LIST LOCAL", list_local)
            # with open(name, 'wb') as handle:
            #     pickle.dump({'boxes': list_local}, handle, protocol=pickle.HIGHEST_PROTOCOL)

            exposures = exposure_class.Exposure(input_ims, srgb_ims, downsample_rate=self.exposureParams["downsample_rate"],
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

    def setValues(self, dummy=1):
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
            self.setAutoExposure()

            # exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"], r_percent=self.exposureParams['r_percent'], g_percent=self.exposureParams['g_percent'],
            #                                     col_num_grids=self.exposureParams['col_num_grids'], row_num_grids=self.exposureParams['row_num_grids'], low_threshold=self.exposureParams['low_threshold'], start_index=self.exposureParams['start_index'],
            #                                     high_threshold=self.exposureParams['high_threshold'], high_rate=self.exposureParams['high_rate'])
            # self.eV,weighted_means,hists,hists_before_ds_outlier = exposures.pipeline()

            # self.img_mean_list = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.defScene.get() + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2 ** self.bit_depth - 1)

            # self.img_mertens = np.load(
            #     os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[
            #         self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')

            self.img_all = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                    self.scene_index] + '_show_dng_imgs' + '.npy')

            self.raw_ims = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays_exposure_224_336') + self.joinPathChar +
                self.scene[
                    self.scene_index] + '_ds_raw_imgs' + '.npy')
            self.blend_rgb()
            # self.img_all = self.img_raw
            button_functions.resetValues(self)
