from copy import deepcopy
from PIL import Image, ImageTk

import constants
import exposure_entropy
import exposure_global
import exposure_saliency
import exposure_semantic
import os
import platform
import button_functions
import manual_semantic_functions
import numpy as np
import tkinter as tk


class Browser:

    def __init__(self, root):
        super().__init__()
        self.AutoExposureList = None
        self.weighted_means = None
        self.eV_original = None
        self.b = None
        self.exposureParams = None
        self.selAutoExposureLabel = None
        self.sceneList = None
        self.selSceneLabel = None
        self.defScene = None
        self.verSlider = None
        self.verSliderLabel = None
        self.horSlider = None
        self.defAutoExposure = None
        self.blended_raw_ims = None
        self.current_auto_exposure = "None"
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
        self.scene_index = 0
        self.temp_img_ind = 0
        self.making_a_serious_of_videos = 0
        self.red_ratio = 0.25
        self.green_ratio = 0.5
        self.blue_ratio = 0.25
        self.joinPathChar = "/"
        if platform.system() == "Windows":
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
        self.video_speed = constants.video_speed
        self.video_fps = constants.video_fps
        self.make_crop_video_flag = 0
        self.temp_coords_for_video_producing = []
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.img).resize((self.imgSize[1], self.imgSize[0])))
        self.canvas = tk.Canvas(root, cursor="cross", width=self.photo.width(), height=self.photo.height(),
                                borderwidth=0, highlightthickness=0)
        self.canvas.grid(row=1, column=1, columnspan=3, rowspan=27, padx=0, pady=0, sticky=tk.NW)
        self.canvas_img = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.current_rects = []  # the rectangles drawn in canvas
        self.rectangles = []  # the coordinates of the rectangles
        self.current_rects_wo_grids = []
        self.rects_without_grids = []  # the coordinates of the rectangles @ local without grids
        self.rects_without_grids_moving_objests = {}  # key: frame number, value: list of rect ids
        self.moving_rectids = []
        self.the_moving_rect = None
        self.the_scrolling_rect = None
        self.rect = None
        self.x = 0
        self.y = 0
        self.start_x = None
        self.start_y = None
        self.curX = 0
        self.curY = 0
        self.num_bins = constants.num_bins
        self.hists = []
        self.fig_2 = None
        self.fig = None
        self.fig_4 = None
        self.local_consider_outliers_check = 0
        self.show_srgb_hist_check = 1
        self.srgb_mean = 0
        self.check = True

    def init_functions(self):
        self.scene_select()
        self.auto_exposure_select()
        self.blend_rgb()

    def blend_rgb(self):
        self.blended_raw_ims = (self.raw_ims[:, :, ::2, 1::2] + self.raw_ims[:, :, 1::2, ::2]) * self.green_ratio / 2 \
                               + self.raw_ims[:, :, ::2, ::2] * self.red_ratio + self.raw_ims[:, :, 1::2,
                                                                                 1::2] * self.blue_ratio

    def horizontal_slider(self, command_function):
        # Horizontal Slider
        self.horSlider = tk.Scale(self.root, activebackground='black', cursor='sb_h_double_arrow', from_=0,
                                  to=self.frame_num[0] - 1,
                                  label='Frame Number', font=(self.widgetFont, self.widgetFontSize),
                                  orient=tk.HORIZONTAL,
                                  length=self.widthToScale, command=command_function)
        self.horSlider.grid(row=27, column=1, columnspan=3, sticky=tk.SW)

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

        self.verSlider.grid(row=1, column=0, rowspan=25)

    def scene_select(self):
        # Select Scene List
        self.defScene = tk.StringVar(self.root)
        self.defScene.set(self.scene[self.scene_index])  # default value
        self.selSceneLabel = tk.Label(self.root, text='Select Scene:', font=(self.widgetFont, self.widgetFontSize))
        self.selSceneLabel.grid(row=0, column=4, sticky=tk.W)
        self.sceneList = tk.OptionMenu(self.root, self.defScene, *self.scene, command=self.setValues)
        self.sceneList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.sceneList.grid(row=1, column=4, sticky=tk.NE)

    def auto_exposure_select(self):
        # Select Scene List
        self.defAutoExposure = tk.StringVar(self.root)
        self.defAutoExposure.set(self.auto_exposures[0])  # default value
        self.selAutoExposureLabel = tk.Label(self.root, text='Select AutoExposure:',
                                             font=(self.widgetFont, self.widgetFontSize))
        self.selAutoExposureLabel.grid(row=0, column=5, sticky=tk.W)
        self.AutoExposureList = tk.OptionMenu(self.root, self.defAutoExposure, *self.auto_exposures,
                                              command=self.setAutoExposure)
        self.AutoExposureList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.AutoExposureList.grid(row=1, column=5, sticky=tk.NE)

    def buttons_builder(self, text, command_function, row, column, para=0):
        self.b = tk.Button(self.root,
                           text=text,
                           fg='#ffffff',
                           bg='#999999',
                           activebackground='#454545',
                           relief=tk.RAISED,
                           width=16,
                           padx=10,
                           pady=5,
                           font=(constants.widgetFont, constants.widgetFontSize),
                           command=lambda: command_function(para))
        self.b.grid(row=row, column=column, sticky=tk.E)

    def setAutoExposure(self, dummy=1):
        self.current_auto_exposure = self.defAutoExposure.get()
        self.scene_index = self.scene.index(self.defScene.get())
        input_ims = self.blended_raw_ims
        self.exposureParams = {
            'low_threshold': self.low_threshold.get(),
            'start_index': float(self.start_index.get()),
            'high_threshold': self.high_threshold.get(),
            "target_intensity": self.target_intensity.get()
        }
        if self.current_auto_exposure == "Global":
            button_functions.clear_rects(self)
            exposures = exposure_global.ExposureGlobal(input_ims,
                                                       target_intensity=self.exposureParams['target_intensity'],
                                                       low_threshold=self.exposureParams['low_threshold'],
                                                       start_index=self.exposureParams['start_index'],
                                                       high_threshold=self.exposureParams['high_threshold'],
                                                       )
            # exposures = exposure_class.Exposure(params = self.exposureParams)

            self.eV, self.eV_original, self.weighted_means, self.hists = exposures.pipeline()

        elif self.current_auto_exposure == "Saliency_map":
            button_functions.clear_rects(self)
            map_name = self.scene[self.scene_index] + "_salient_maps_mbd.npy"
            salient_map_mbd = np.load("saliency_maps/" + map_name)
            salient_map = salient_map_mbd
            exposures = exposure_saliency.ExposureSaliency(input_ims, salient_map,
                                                           target_intensity=self.exposureParams['target_intensity'],
                                                           low_threshold=self.exposureParams['low_threshold'],
                                                           start_index=self.exposureParams['start_index'],
                                                           high_threshold=self.exposureParams['high_threshold'],

                                                           )
            # exposures = exposure_class.Exposure(params = self.exposureParams)

            self.eV, self.eV_original, self.weighted_means, self.hists = exposures.pipeline()

        elif self.current_auto_exposure == "Entropy":
            button_functions.clear_rects(self)
            srgb_ims = self.img_all[:, :, ::8, ::8, :]
            exposures = exposure_entropy.ExposureEntropy(input_ims,
                                                         srgb_ims,
                                                         start_index=self.exposureParams['start_index'],
                                                         )
            # exposures = exposure_class.Exposure(params = self.exposureParams)
            self.eV, self.eV_original, self.weighted_means, self.hists = exposures.pipeline()

        elif self.current_auto_exposure == "Semantic":
            button_functions.clear_rects_local(self)
            list_local = manual_semantic_functions.list_local_without_grids_moving_objects(self)
            exposures = exposure_semantic.ExposureSemantic(input_ims, list_local,
                                                           target_intensity=self.exposureParams['target_intensity'],
                                                           low_threshold=self.exposureParams['low_threshold'],
                                                           start_index=self.exposureParams['start_index'],
                                                           high_threshold=self.exposureParams['high_threshold'],
                                                           )
            self.eV, self.eV_original, self.weighted_means, self.hists = exposures.pipeline()

    def setValues(self, dummy=1):
        button_functions.runVideo(self)
        if self.scene[self.scene_index] != self.defScene.get():
            button_functions.clear_rects(self)
            self.scene_index = self.scene.index(self.defScene.get())
            self.setAutoExposure()
            self.img_all = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                    self.scene_index] + '_show_dng_imgs' + '.npy')

            self.raw_ims = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays_exposure_224_336') + self.joinPathChar +
                self.scene[
                    self.scene_index] + '_ds_raw_imgs' + '.npy')
            self.blend_rgb()
            button_functions.resetValues(self)
