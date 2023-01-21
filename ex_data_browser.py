import tkinter
import tkinter as tk

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


mp.rcParams.update({'axes.titlesize': 14, 'font.size': 11, 'font.family': 'arial'})

# Tkinter Window
root = tk.Tk()
root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5


class Browser:

    def __init__(self, root):
        super().__init__()


        self.folders = "E:\Final"  # link to directory containing all the dataset image folders

        self.widgetFont = 'Arial'
        self.widgetFontSize = 12

        # self.scene = ['Scene101', 'Scene102', 'Scene103', 'Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
        #               'Scene7', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15', 'Scene16', 'Scene17', 'Scene18']
        # self.frame_num = [90, 65, 15, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # number of frames per position
        # self.stack_size = [12, 47, 28, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]  # number of shutter options per position

        self.scene = ['Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
                      'Scene7', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15',
                      'Scene16', 'Scene17', 'Scene18', 'Scene19', 'Scene20', 'Scene21','Scene22']
        self.frame_num = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                          100, 100, 100, 100, 100,100]  # number of frames per position
        self.stack_size = [40, 15, 40, 15, 15, 40, 15, 15, 15, 40, 40, 15, 15, 15, 40, 40, 40,
                           40, 40, 40, 40, 40]  # number of shutter options per position

        self.SCALE_LABELS = {
            0: '15"',
            1: '8"',
            2: '6"',
            3: '4"',
            4: '2"',
            5: '1"',
            6: '0"5',
            7: '1/4',
            8: '1/8',
            9: '1/15',
            10: '1/30',
            11: '1/60',
            12: '1/125',
            13: '1/250',
            14: '1/500'
        }
        self.SCALE_LABELS_NEW = {
            0: '15"',
            1: '13"',
            1: '13"',
            2: '10"',
            3: '8"',
            4: '6"',
            5: '5"',
            6: '4"',
            7: '3"2',
            8: '2"5',
            9: '2"',
            10: '1"6',
            11: '1"3',
            12: '1"',
            13: '0"8',
            14: '0"6',
            15: '0"5',
            16: '0"4',
            17: '0"3',
            18: '1/4',
            19: '1/5',
            20: '1/6',
            21: '1/8',
            22: '1/10',
            23: '1/13',
            24: '1/15',
            25: '1/20',
            26: '1/25',
            27: '1/30',
            28: '1/40',
            29: '1/50',
            30: '1/60',
            31: '1/80',
            32: '1/100',
            33: '1/125',
            34: '1/160',
            35: '1/200',
            36: '1/250',
            37: '1/320',
            38: '1/400',
            39: '1/500'
        }
        self.NEW_SCALES = [15,13,10,8,6,5,4,3.2,2.5,2,1.6,1.3,1,0.8,0.6,0.5,0.4,0.3,1/4,1/5,1/6,1/8,1/10,1/13,1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100,1/125,1/160,1/200,1/250,1/320,1/400,1/500]

        self.eV = []
        self.auto_exposures = ["None", "Global","Saliency_map", "Local", 'Local without grids', 'Local on moving objects','Max Gradient srgb','Entropy','HDR Histogram Method']
        self.current_auto_exposure = "None"

        self.scene_index = 18
        self.mertensVideo = []
        self.bit_depth = 8
        self.downscale_ratio = 0.12
        self.check = True
        self.temp_img_ind = 0
        self.making_a_serious_of_videos = 0

        self.joinPathChar = "/"
        if (platform.system() == "Windows"):
            self.joinPathChar = "\\"

        self.imgSize = [int(4480 * self.downscale_ratio), int(6720 * self.downscale_ratio)]
        self.widthToScale = self.imgSize[1]
        self.widPercent = (self.widthToScale / float(self.imgSize[1]))
        self.heightToScale = int(float(self.imgSize[0]) * float(self.widPercent))

        self.img_all = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[
            self.scene_index] + '_imgs_' + str(self.downscale_ratio) + '.npy')
        self.img_mean_list = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[
                self.scene_index] + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2 ** self.bit_depth - 1)
        self.img_mertens = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[
                self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')
        self.img_raw = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                self.scene_index] + '_show_dng_imgs' + '.npy')
        # if self.stack_size[self.scene_index] == 40:
        #     self.img_all = self.img_raw
        self.img = deepcopy(self.img_all[0])
        self.useMertens = False
        self.useRawIms = 1
        self.play = True
        self.video_speed = 50
        self.video_fps = 30

        self.res_check = 0
        self.hdr_mode_check = 0
        self.make_crop_video_flag = 0
        self.temp_coords_for_video_producing = []

        # Image Convas
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.img))

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
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_move_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.canvas.bind("<Button-3>", self.right_click)
        self.canvas.bind("<MouseWheel>", self.zoomer)
        self.canvas.bind("<Button-4>", self.zoomerP)
        self.canvas.bind("<Button-5>", self.zoomerM)
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
        self.num_bins = 100
        self.hists = []
        self.hists_before_ds_outlier = []

        self.fig_2 = None
        self.fig = None
        self.fig_4 = None
        self.local_consider_outliers_check = 0
        self.show_srgb_hist_check = 0
        self.srgb_mean = 0
        self.init_functions()
        self.show_srgb_hist_check = self.show_srgb_hist_check_.get()

    def init_functions(self):

        # self.hdr_mean_button()
        # self.hdr_median_button()
        # self.hdr_mertens_button()
        # self.hdr_abdullah_button()
        self.hdr_run_button()
        self.hdr_pause_button()
        self.hdr_reset_button()
        self.scene_select()
        self.playback_text_box()
        self.video_fps_text()
        self.horizontal_slider()
        self.vertical_slider()
        self.target_intensity_text_box()
        self.image_mean_plot()
        self.hist_plot()
        self.hist_plot_three(stack_size=15, curr_frame_mean_list=np.zeros(15))
        self.regular_video_button()
        self.high_res_checkbox()
        self.mertens_checkbox()
        self.outlier_slider()
        self.col_num_grids_text()
        self.row_num_grids_text()
        self.show_Raw_Ims_check_box()
        self.clear_interested_areas_button()
        self.local_consider_outliers_checkbox()
        self.auto_exposure_select()
        self.number_of_previous_frames_text_box()
        self.stepsize_limit_text_box()
        self.save_interested_moving_objects_button()
        self.make_global_videos_button()
        self.make_local_videos_button()
        self.make_moving_object_videos_button()
        self.local_interested_name_text_box()
        self.local_interested_global_area_percentage_box()
        self.show_SRGB_hist_check_box()

    def local_interested_name_text_box(self):
        self.local_interested_name = tk.StringVar()
        self.local_interested_name.set("")
        tk.Label(root, text="Name of Interested").grid(row=10, column=5)
        self.e1 = tk.Entry(root, textvariable=self.local_interested_name)
        self.e1.grid(row=11, column=5, sticky=tk.E)

    def target_intensity_text_box(self):
        self.target_intensity = tk.DoubleVar()
        self.target_intensity.set(0.13)
        tk.Label(root, text="target intensity").grid(row=29, column=5)
        self.e1 = tk.Entry(root, textvariable=self.target_intensity)
        self.e1.grid(row=30, column=5, sticky=tk.E)



    def local_interested_global_area_percentage_box(self):
        self.local_interested_global_area_percentage = tk.DoubleVar()
        self.local_interested_global_area_percentage.set(0.0)
        tk.Label(root, text="Global percentage on local selection").grid(row=25, column=5)
        self.e1 = tk.Entry(root, textvariable=self.local_interested_global_area_percentage)
        self.e1.grid(row=26, column=5, sticky=tk.E)

    def make_global_videos_button(self):
        self.makeGlobalVideosButton = tk.Button(root, text='Make Global Videos', fg='#ffffff', bg='#999999',
                                                activebackground='#454545',
                                                relief=tk.RAISED, width=16, padx=10, pady=5,
                                                font=(self.widgetFont, self.widgetFontSize),
                                                command=self.make_global_videos)
        self.makeGlobalVideosButton.grid(row=11 - 4, column=5, sticky=tk.E)

    def make_local_videos_button(self):
        self.makeLocalVideosButton = tk.Button(root, text='Make Local Videos', fg='#ffffff', bg='#999999',
                                               activebackground='#454545',
                                               relief=tk.RAISED, width=16, padx=10, pady=5,
                                               font=(self.widgetFont, self.widgetFontSize),
                                               command=self.make_local_videos)
        self.makeLocalVideosButton.grid(row=12 - 4, column=5, sticky=tk.E)

    def make_moving_object_videos_button(self):
        self.makeMovingObjectVideosButton = tk.Button(root, text='Moving Object Videos', fg='#ffffff', bg='#999999',
                                               activebackground='#454545',
                                               relief=tk.RAISED, width=16, padx=10, pady=5,
                                               font=(self.widgetFont, self.widgetFontSize),
                                               command=self.make_moving_object_videos)
        self.makeMovingObjectVideosButton.grid(row=13 - 4, column=5, sticky=tk.E)

    def save_interested_moving_objects_fuction(self):
        if self.current_auto_exposure == "Local on moving objects" and len(self.moving_rectids) > 0:
            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            curr_frame = self.horSlider.get()
            temp = []
            for id in self.moving_rectids:
                coor = self.canvas.coords(id)
                temp.append([coor[1] / h, coor[0] / w, coor[3] / h, coor[2] / w])
            self.rects_without_grids_moving_objests[curr_frame] = temp.copy()
            # self.moving_rectids = []
            print("the dict of interests")
            print(self.rects_without_grids_moving_objests)

    def draw_interested_moving_areas_per_frame(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        curr_frame = self.horSlider.get()
        print("length of moving areas")
        print(len(self.the_moving_area_list))
        try:
            rect_coordis = self.the_moving_area_list[curr_frame]
            self.clear_moving_rects()
            for coords in rect_coordis:
                a,b,c,d = coords[1]*w, coords[0]*h, coords[3]*w, coords[2]*h
                rect = self.canvas.create_rectangle(a, b, c, d, outline='violet')
                self.moving_rectids.append(rect)
        except:
            pass



    def save_interested_moving_objects_button(self):
        self.movingObjectButton = tk.Button(root, text='Save Interested Area', fg='#ffffff', bg='#999999',
                                            activebackground='#454545',
                                            relief=tk.RAISED, width=16, padx=10, pady=5,
                                            font=(self.widgetFont, self.widgetFontSize),
                                            command=self.save_interested_moving_objects_fuction)
        self.movingObjectButton.grid(row=10 - 4, column=5,
                                     sticky=tk.E)  # initial row was 26, +1 increments for all other rows

    def hdr_mean_button(self):
        # HDR Button - Mean
        self.HdrMeanButton = tk.Button(root, text='HDR-Mean', fg='#ffffff', bg='#999999', activebackground='#454545',
                                       relief=tk.RAISED, width=16, padx=10, pady=5,
                                       font=(self.widgetFont, self.widgetFontSize), command=self.HdrMean)
        self.HdrMeanButton.grid(row=1, column=5, sticky=tk.E)  # initial row was 26, +1 increments for all other rows

    def hdr_median_button(self):
        # HDR Button - Median
        self.HdrMedianButton = tk.Button(root, text='HDR-Median', fg='#ffffff', bg='#999999',
                                         activebackground='#454545',
                                         relief=tk.RAISED, width=16, padx=10, pady=5,
                                         font=(self.widgetFont, self.widgetFontSize), command=self.HdrMedian)
        self.HdrMedianButton.grid(row=2, column=5, sticky=tk.E)

    def hdr_mertens_button(self):
        # HDR Button - Mertens
        self.HdrMertensButton = tk.Button(root, text='HDR-Mertens', fg='#ffffff', bg='#999999',
                                          activebackground='#454545',
                                          relief=tk.RAISED, width=16, padx=10, pady=5,
                                          font=(self.widgetFont, self.widgetFontSize), command=self.HdrMertens)
        self.HdrMertensButton.grid(row=3, column=5, sticky=tk.E)

    def show_Raw_Ims_check_box(self):
        self.useRawIms_ = tk.IntVar()
        self.useRawIms_.set(1)
        self.c1 = tk.Checkbutton(root, text='Show Raw Image', variable=self.useRawIms_, offvalue=0, onvalue=1,
                                 command=self.switch_raw)
        self.c1.grid(row=24, column=5)

    def show_SRGB_hist_check_box(self):
        self.show_srgb_hist_check_ = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text='Show SRGB Hist', variable=self.show_srgb_hist_check_, offvalue=0, onvalue=1,
                                 command=self.switch_SRGB_Hist)
        self.c1.grid(row=27, column=5)
    def hdr_abdullah_button(self):
        # HDR Button - Abdullah
        self.HdrAbdullahButton = tk.Button(root, text='HDR-Abdullah', fg='#ffffff', bg='#999999',
                                           activebackground='#454545',
                                           relief=tk.RAISED, width=16, padx=10, pady=5,
                                           font=(self.widgetFont, self.widgetFontSize),
                                           command=self.HdrAbdullah)
        self.HdrAbdullahButton.grid(row=4, column=5, sticky=tk.E)

    def hdr_run_button(self):
        # Run Button
        self.RunButton = tk.Button(root, text='Run', fg='#ffffff', bg='#999999', activebackground='#454545',
                                   relief=tk.RAISED, width=16, padx=10, pady=5,
                                   font=(self.widgetFont, self.widgetFontSize), command=self.runVideo)
        self.RunButton.grid(row=5 - 4, column=5, sticky=tk.E)

    def hdr_pause_button(self):
        self.PauseButton = tk.Button(root, text='Pause', fg='#ffffff', bg='#999999', activebackground='#454545',
                                     relief=tk.RAISED, padx=10, pady=5,
                                     width=16, font=(self.widgetFont, self.widgetFontSize), command=self.pauseRun)
        self.PauseButton.grid(row=6 - 4, column=5, sticky=tk.E)

    def hdr_reset_button(self):
        # Reset Button
        self.RestButton = tk.Button(root, text='Reset', fg='#ffffff', bg='#999999', activebackground='#454545',
                                    relief=tk.RAISED, padx=10, pady=5, width=16,
                                    font=(self.widgetFont, self.widgetFontSize), command=self.resetValues)
        self.RestButton.grid(row=7 - 4, column=5, sticky=tk.E)

    def number_of_previous_frames_text_box(self):
        self.number_of_previous_frames = tk.IntVar()
        self.number_of_previous_frames.set(3)
        tk.Label(root, text="# of previous frames").grid(row=31, column=5)
        self.e1 = tk.Entry(root, textvariable=self.number_of_previous_frames)
        self.e1.grid(row=32, column=5, sticky=tk.E)

    def stepsize_limit_text_box(self):
        self.stepsize_limit = tk.IntVar()
        self.stepsize_limit.set(5)
        tk.Label(root, text="step size limitation").grid(row=33, column=5)
        self.e1 = tk.Entry(root, textvariable=self.stepsize_limit)
        self.e1.grid(row=34, column=5, sticky=tk.E)

    def regular_video_button(self):

        self.VideoButton = tk.Button(root, text='Video', fg='#ffffff', bg='#999999', activebackground='#454545',
                                     relief=tk.RAISED, padx=10, pady=5,
                                     width=16, font=(self.widgetFont, self.widgetFontSize), command=self.export_video)
        self.VideoButton.grid(row=8 - 4, column=5, sticky=tk.E)

    def outlier_slider(self):
        self.low_threshold = tk.DoubleVar()
        self.high_threshold = tk.DoubleVar()
        #print("BUILDING OUTLIER SLIDER", self.high_threshold)
        self.outlierSlider = RangeSliderH(root, [self.low_threshold, self.high_threshold], Width=400, Height=65,
                                          min_val=0, max_val=0.9, show_value=True, padX=25
                                          , line_s_color="#7eb1c2", digit_precision='.2f')
        #self.high_threshold.set(0.95)
        #self.outlierSlider.
        #print("BUILDING OUTLIER SLIDER", self.high_threshold)

        self.outlierSlider.grid(padx=10, pady=10, row=28, column=2, columnspan=1, sticky=tk.E)
        # self.show_threshold()
        self.start_index_text_box()
        self.high_rate_text_box()

    def start_index_text_box(self):
        self.start_index = tk.StringVar()
        self.start_index.set("15")
        tk.Label(root, text="start index").grid(row=29, column=2)
        self.e1 = tk.Entry(root, textvariable=self.start_index)
        self.e1.grid(row=30, column=2)

    def high_rate_text_box(self):
        self.high_rate = tk.StringVar()
        self.high_rate.set("0.2")
        tk.Label(root, text="above high threshold").grid(row=31, column=2)
        self.e1 = tk.Entry(root, textvariable=self.high_rate)
        self.e1.grid(row=32, column=2)

    # def show_threshold(self):
    #     print(self.low_threshold.get())
    #     print(self.high_threshold.get())

    def clear_interested_areas_button(self):
        # clear the rects
        self.ClearInterestedAreasButton = tk.Button(root, text='Clear Rectangles', fg='#ffffff', bg='#999999',
                                                    activebackground='#454545',
                                                    relief=tk.RAISED, width=16, padx=10, pady=5,
                                                    font=(self.widgetFont, self.widgetFontSize),
                                                    command=self.clear_rects,
                                                    )
        self.ClearInterestedAreasButton.grid(row=9 - 4, column=5,
                                             sticky=tk.E)  # initial row was 26, +1 increments for all other rows

    def scene_select(self):
        # Select Scene List
        self.defScene = tk.StringVar(root)
        self.defScene.set(self.scene[self.scene_index])  # default value
        self.selSceneLabel = tk.Label(root, text='Select Scene:', font=(self.widgetFont, self.widgetFontSize))
        self.selSceneLabel.grid(row=0, column=3, sticky=tk.W)
        self.sceneList = tk.OptionMenu(root, self.defScene, *self.scene, command=self.setValues)
        self.sceneList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.sceneList.grid(row=1, column=3, sticky=tk.NE)

    def auto_exposure_select(self):
        # Select Scene List
        self.defAutoExposure = tk.StringVar(root)
        self.defAutoExposure.set(self.auto_exposures[0])  # default value
        self.selAutoExposureLabel = tk.Label(root, text='Select AutoExposure:',
                                             font=(self.widgetFont, self.widgetFontSize))
        self.selAutoExposureLabel.grid(row=0, column=4, sticky=tk.W)
        self.AutoExposureList = tk.OptionMenu(root, self.defAutoExposure, *self.auto_exposures,
                                              command=self.setAutoExposure)
        self.AutoExposureList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.AutoExposureList.grid(row=1, column=4, sticky=tk.NE)

    def playback_text_box(self):
        # TextBox
        self.video_speed = tk.StringVar()
        # video_speed = 1
        tk.Label(root, text="Browser Playback Speed (ms delay)").grid(row=30, column=1)
        self.e1 = tk.Entry(root, textvariable=self.video_speed)
        self.e1.grid(row=31, column=1)

    def video_fps_text(self):
        # TextBox
        self.save_video_fps = tk.StringVar()
        # video_speed = 1
        tk.Label(root, text="Video FPS").grid(row=32, column=1)
        self.e1 = tk.Entry(root, textvariable=self.save_video_fps)
        self.e1.grid(row=33, column=1)

    def col_num_grids_text(self):
        # TextBox
        self.col_num_grids_ = tk.StringVar()
        tk.Label(root, text=" Number of Grids per Column").grid(row=19, column=5)
        self.col_num_grids_.set('8')
        self.e1 = tk.Entry(root, textvariable=self.col_num_grids_)
        self.e1.grid(row=20, column=5)

    def row_num_grids_text(self):
        # TextBox
        self.row_num_grids_ = tk.StringVar()
        self.row_num_grids_.set('8')
        tk.Label(root, text=" Number of Grids per Row").grid(row=21, column=5)
        self.e1 = tk.Entry(root, textvariable=self.row_num_grids_)
        self.e1.grid(row=22, column=5)

    def local_consider_outliers_checkbox(self):
        self.local_consider_outliers_check_ = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text='Consider outliers at local selection',
                                 variable=self.local_consider_outliers_check_, offvalue=0, onvalue=1,
                                 command=self.switch_outlier)
        self.c1.grid(row=23, column=5)

    def high_res_checkbox(self):
        self.high_res_check = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text='High Resolution', variable=self.high_res_check, offvalue=0, onvalue=1,
                                 command=self.switch_res)
        self.c1.grid(row=28, column=1)

    def switch_outlier(self):
        self.local_consider_outliers_check = self.local_consider_outliers_check_.get()
        print("local_consider_outliers_check is ", self.local_consider_outliers_check)

    def switch_res(self):

        self.res_check = self.high_res_check.get()
        print("self.res_check is ", self.res_check)

    def switch_raw(self):

        self.useRawIms = self.useRawIms_.get()
        print("self.useRawIms is ", self.useRawIms)
        self.updateSlider(0)

    def switch_SRGB_Hist(self):

        self.show_srgb_hist_check = self.show_srgb_hist_check_.get()
        print("self.show_srgb_hist_chec is ", self.show_srgb_hist_check)
        self.updateSlider(0)

    def mertens_checkbox(self):

        self.mertens_check = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text=' Mertens Export', variable=self.mertens_check, offvalue=0, onvalue=1,
                                 command=self.switch_mertens)
        self.c1.grid(row=29, column=1)

    def switch_mertens(self):

        self.hdr_mode_check = self.mertens_check.get()
        print("self.hdr_mode_check is ", self.hdr_mode_check)

    def horizontal_slider(self):
        # Horizantal Slider
        self.horSlider = tk.Scale(root, activebackground='black', cursor='sb_h_double_arrow', from_=0,
                                  to=self.frame_num[0] - 1,
                                  label='Frame Number', font=(self.widgetFont, self.widgetFontSize),
                                  orient=tk.HORIZONTAL,
                                  length=self.widthToScale, command=self.updateSlider)
        self.horSlider.grid(row=27, column=1, columnspan=2, sticky=tk.SW)

    def vertical_slider(self):
        # Vertical Slider

        self.verSliderLabel = tk.Label(root, text='Exposure Time', font=(self.widgetFont, self.widgetFontSize))
        self.verSliderLabel.grid(row=0, column=0)

        # self.verSlider = tk.Scale(root, activebackground='black', cursor='sb_v_double_arrow', from_=0,
        #                      to=self.stack_size[self.scene_index] - 1, font=(self.widgetFont, self.widgetFontSize), length=self.heightToScale,
        #                      command=self.updateSlider)

        if self.stack_size[self.scene_index] == 40:
            min_ = min(self.SCALE_LABELS_NEW)
            max_ = max(self.SCALE_LABELS_NEW)
            min_ = 0
            max_ = len(self.SCALE_LABELS_NEW) - 1
        else:
            min_ = min(self.SCALE_LABELS)
            max_ = max(self.SCALE_LABELS)
            min_ = 0
            max_ = len(self.SCALE_LABELS) - 1
        max_ = 40
        self.verSlider = tk.Scale(root, activebackground='black', cursor='sb_v_double_arrow', from_=min_, to=max_,
                                  font=(self.widgetFont, self.widgetFontSize),
                                  length=self.heightToScale,
                                  command=self.scale_labels)

        # print(self.verSlider.configure().keys())

        self.verSlider.grid(row=1, column=0, rowspan=25)

    def scale_labels(self, value):

        # self.verSlider.config(label=self.SCALE_LABELS[int(value)])
        if self.stack_size[self.scene_index] == 40:
            text_ = self.SCALE_LABELS_NEW[int(value)]
        else:
            text_ = self.SCALE_LABELS[int(value)]
        tk.Label(root, text=text_, font=("Times New Roman", 15)).grid(row=27, column=0, )

        # self.verSlider.place(x=50, y=300, anchor="center")
        self.useMertens = False

        # if(self.current_auto_exposure != "None"):
        #     self.check = True
        temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + int(self.verSlider.get())

        self.updateHorSlider(value, temp_img_ind)

        # scale = tk.Scale(root, from_=min(SCALE_LABELS), to=max(SCALE_LABELS),
        #                  orient=tk.HORIZONTAL, showvalue=False, command=scale_labels)

    def image_mean_plot(self, ind=0, val=0):
        stack_size = self.stack_size[self.scene_index]
        curr_frame_mean_list = np.zeros(stack_size)
        if self.fig:
            plt.close(self.fig)
            self.fig.clear()
        self.fig = plt.figure(figsize=(4, 3))  # 4.6, 3.6
        # plt.plot(np.arange(self.stack_size[self.scene_index]), self.img_mean_list[0:self.stack_size[self.scene_index]], color='green',
        #          linewidth=2)  # ,label='Exposure stack mean')
        # plt.plot(0, self.img_mean_list[0], color='red', marker='o', markersize=12)
        # plt.text(0, self.img_mean_list[0], '(' + str(0) + ', ' + str("%.2f" % self.img_mean_list[0]) + ')', color='red',
        #          fontsize=13, position=(0 - 0.2, self.img_mean_list[0] + 0.04))
        plt.plot(np.arange(stack_size), curr_frame_mean_list, color='green',
                 linewidth=2)  # ,label='Exposure stack mean')
        plt.plot(ind, val, color='red', marker='o', markersize=12)
        plt.text(ind, val, '(' + str(ind) + ', ' + str("%.2f" % val) + ')', color='red',
                 fontsize=13, position=(0 - 0.2, val + 0.04))
        plt.title('Exposure stack mean')
        plt.xlabel('Image index')
        plt.ylabel('Mean value')
        plt.xlim(-0.2, stack_size - 0.8)
        plt.xticks(np.arange(0, stack_size, 1))
        # if stack_size < 20:
        #     plt.xticks(np.arange(0, stack_size, 1))
        # elif stack_size >= 15 and stack_size < 30:
        #     plt.xticks(np.arange(0, stack_size, 2))
        # else:
        #     plt.xticks(np.arange(0, stack_size, 3))

        plt.ylim(-0.02, 0.85)
        plt.yticks(np.arange(0, 0.85, 0.1))
        self.fig.canvas.draw()

        self.tempImg_2 = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        self.photo_2 = ImageTk.PhotoImage(self.tempImg_2)
        self.imagePrevlabel_2 = tk.Label(root, image=self.photo_2)
        self.imagePrevlabel_2.grid(row=2, column=3, columnspan=2, rowspan=15, sticky=tk.NE)

    def hist_plot(self, count1=np.zeros(101), count2=np.zeros(101)):
        font = {'family': 'monospace',
                'weight': 'bold',
                'size': 10}
        bins = np.arange(1, self.num_bins + 2)
        # self.fig = plt.figure(figsize=(4, 4))  # 4.6, 3.6
        if self.fig_2:
            plt.close(self.fig_2)
            self.fig_2.clear()
        self.fig_2, axes = plt.subplots(2, sharex=True, sharey=True, figsize=(4, 6))

        axes[1].bar(bins, count2, align='center')
        axes[0].bar(bins, count1, align='center')
        if self.show_srgb_hist_check == 1:
            axes[1].set_title('srgb histogram', **font)
        else:
            axes[1].set_title('histogram with outlier', **font)
        axes[0].set_title('histogram without outlier', **font)

        self.fig_2.canvas.draw()

        self.tempImg_3 = Image.frombytes('RGB', self.fig_2.canvas.get_width_height(), self.fig_2.canvas.tostring_rgb())
        self.photo_3 = ImageTk.PhotoImage(self.tempImg_3)
        self.imagePrevlabel_3 = tk.Label(root, image=self.photo_3)
        self.imagePrevlabel_3.grid(row=17, column=3, columnspan=2, rowspan=20, sticky=tk.NE)

    def hist_plot_three(self, stack_size, curr_frame_mean_list, count1=np.zeros(101), count2=np.zeros(101),
                        count3=np.zeros(101), ind=0, val=0, ind2=0, val2=0):
        # stack_size = self.stack_size[self.scene_index]
        # curr_frame_mean_list = np.zeros(stack_size)
        font = {'family': 'monospace',
                'weight': 'bold',
                'size': 10}
        bins = np.arange(1, self.num_bins + 2)
        # self.fig = plt.figure(figsize=(4, 4))  # 4.6, 3.6
        if self.fig_2:
            plt.close(self.fig_2)
            self.fig_2.clear()
        self.fig_2, axes = plt.subplots(3, figsize=(4, 9))
        self.fig_2.tight_layout()
        sum_c1 = max(sum(count1), 1)

        # print(count1)
        # print("=========")
        # print(count2)
        sum_c2 = max(sum(count2), 1)
        sum_c3 = max(sum(count3), 1)
        vals1 = count1 / sum_c1
        vals2 = count2 / sum_c2
        vals3 = count3 / sum_c3
        if ind == ind2:
            color1 = 'blue'
            axes[1].bar(bins, vals2, align='center', color=color1)
            if self.show_srgb_hist_check == 1:
                axes[1].set_title('srgb histogram', **font)
                axes[1].text(70, 0.2, '( mean: ' + str("%.2f" % self.srgb_mean) + ')', color='blue',
                         fontsize=13,position=(70,0.2))
            else:
                axes[1].set_title('histogram with outlier', **font)
            axes[0].set_title('histogram without outlier', **font)
            for i, x in enumerate(vals2):
                if x > 0.25:
                    axes[1].text(i, 0.25, str("%.2f" % x), color=color1,
                                 fontsize=13, position=(i, 0.251))
        else:
            color1 = 'orange'
            axes[1].bar(bins, vals3, align='center', color=color1)
            axes[1].set_title('selected image histogram', **font)
            axes[0].set_title('current image histogram', **font)
            for i, x in enumerate(vals3):
                if x > 0.25:
                    axes[1].text(i, 0.25, str("%.2f" % x), color=color1,
                                 fontsize=13, position=(i, 0.251))

        axes[0].bar(bins, vals1, align='center', color='violet')
        axes[0].set_ylim([0, 0.25])
        axes[1].sharex(axes[0])
        axes[1].sharey(axes[0])
        for i, x in enumerate(vals1):
            if x > 0.25:
                axes[0].text(i, 0.25, str("%.2f" % x), color='violet',
                             fontsize=13, position=(i, 0.251))

        axes[2].plot(np.arange(stack_size), curr_frame_mean_list, color='green',
                     linewidth=2)  # ,label='Exposure stack mean')
        axes[2].plot(ind, val, color='violet', marker='o', markersize=12)
        axes[2].text(ind, val, '(' + str(ind) + ', ' + str("%.2f" % val) + ')', color='violet',
                     fontsize=13, position=(ind - 0.2, val + 0.01))
        if ind != ind2:
            axes[2].plot(ind2, val2, color='orange', marker='o', markersize=12)
            axes[2].text(ind2, val2, '(' + str(ind2) + ', ' + str("%.2f" % val2) + ')', color='orange',
                         fontsize=13, position=(ind2 - 0.2, val2 + 0.01))
        axes[2].set_title('Exposure stack mean', **font)
        axes[2].set_ylim([-0.1, 1.1])
        axes[2].set_xlim(-1, stack_size)

        axes[2].set_xticks(np.arange(0, stack_size, 2))

        self.fig_2.canvas.draw()

        self.tempImg_3 = Image.frombytes('RGB', self.fig_2.canvas.get_width_height(),
                                         self.fig_2.canvas.tostring_rgb())
        self.photo_3 = ImageTk.PhotoImage(self.tempImg_3)
        self.imagePrevlabel_3 = tk.Label(root, image=self.photo_3)
        self.imagePrevlabel_3.grid(row=2, column=3, columnspan=2, rowspan=45, sticky=tk.NE)

    def HdrMean(self):

        temp_img_ind = int(self.horSlider.get() * self.stack_size[self.scene_index])
        temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])

        temp_img = (np.mean(temp_stack, axis=0)).astype(np.uint8)
        cv2.putText(temp_img, 'HDR-Mean', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # tempImg = Image.fromarray(temp_img)
        # self.photo = ImageTk.PhotoImage(tempImg)
        # self.imagePrevlabel.configure(image=self.photo)

    def HdrMedian(self):

        temp_img_ind = int(self.horSlider.get() * self.stack_size[self.scene_index])
        temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])

        temp_img = (np.median(temp_stack, axis=0)).astype(np.uint8)
        cv2.putText(temp_img, 'HDR-Median', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # tempImg = Image.fromarray(temp_img)
        # self.photo = ImageTk.PhotoImage(tempImg)

    # self.imagePrevlabel.configure(image=self.photo)

    def HdrMertens(self):

        self.mertensVideo = []
        self.useMertens = True
        self.mertens_pic = []

        self.updateSlider(0)

    def HdrAbdullah(self):

        temp_img_ind = int(self.horSlider.get() * self.stack_size[self.scene_index])
        temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])

        print(temp_img_ind, temp_img_ind + self.stack_size[self.scene_index])

        temp_stack_clip_weights = np.where(temp_stack[:, :, :, 1] < 10, 0.1, 1)
        mean_2 = np.average(temp_stack[:, :, :, 1], axis=0, weights=temp_stack_clip_weights)
        stack_distance = abs(temp_stack[:, :, :, 1] - mean_2)
        stack_distance_arrs = np.argsort(stack_distance, axis=0)
        stack_distance_min = (np.mean(stack_distance_arrs[0:7], axis=0)).astype(np.uint8)
        stack_distance_min_med = cv2.medianBlur(stack_distance_min, 91)
        stack_distance_min_med = cv2.medianBlur(stack_distance_min_med, 151)
        stack_distance_min_med = cv2.medianBlur(stack_distance_min_med, 191)
        mean_min_dis_med = (np.zeros((temp_stack.shape[1], temp_stack.shape[2], temp_stack.shape[3]))).astype(np.uint8)
        for i in range(mean_min_dis_med.shape[0]):
            for j in range(mean_min_dis_med.shape[1]):
                mean_min_dis_med[i, j, :] = temp_stack[stack_distance_min_med[i, j], i, j, :]
        cv2.putText(mean_min_dis_med, 'HDR-Abdullah', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # tempImg = Image.fromarray(mean_min_dis_med)
        # self.photo = ImageTk.PhotoImage(tempImg)
        # self.imagePrevlabel.configure(image=self.photo)

    def export_video(self):

        reg_vid = []
        reg_vid_plot = []
        list = ['15', '8', '6', '4', '2', '1', '05', '1-4', '1-8', '1-15', '1-30', '1-60', '1-125', '1-250', '1-500']

        self.mertensVideo = []
        self.mertens_pic = []

        if self.res_check == 0 and self.current_auto_exposure == "None":

            for i in range(100):
                self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + int(self.verSlider.get())
                self.check = False
                self.updatePlot()
                reg_vid_plot.append(self.tempImg_2)

                img = deepcopy(self.img_all[self.temp_img_ind])
                reg_vid.append(img)
                print("IMG", img.shape, i, "STACK SIZE", self.stack_size[self.scene_index])

            m1 = Image.fromarray(reg_vid[0])
            m2 = reg_vid_plot[0]
            sv = self.get_concat_h_blank(m1, m2)

            self.check_fps()

            fold_name = self.scene[self.scene_index] + "_0.12_Ex_" + list[int(self.verSlider.get())] + "_FPS_" + str(
                self.video_fps)
            folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
            os.makedirs(folderStore, exist_ok=True)
            connected_image = folderStore + self.joinPathChar + fold_name + ".avi"

            # capture the image and save it on the save path
            os.makedirs(folderStore, exist_ok=True)

            video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.video_fps,
                                    (sv.width, sv.height))

            for i in range(len(reg_vid)):
                tempImg = Image.fromarray(reg_vid[i])
                temp_img_plot = reg_vid_plot[i]
                array = np.array(self.get_concat_h_blank(tempImg, temp_img_plot))
                video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

            video.release()

        elif self.res_check == 0 and self.current_auto_exposure != "None":
            if not len(self.eV) == 100:
                return

            for i in range(100):
                self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + self.eV[i]
                self.check = False
                self.updatePlot()
                # reg_vid_plot.append(self.tempImg_2)

                # img = deepcopy(self.img_all[self.temp_img_ind])
                img = deepcopy(self.img_raw[i][self.eV[i]])
                print("IMG", img.shape)

                reg_vid.append(img)


            m1 = Image.fromarray(reg_vid[0])
            # m2 = reg_vid_plot[0]
            # sv = self.get_concat_h_blank(m1, m2)
            sv = m1

            self.check_fps()

            fold_name = self.scene[self.scene_index] + "_dng_pipeline_" + self.current_auto_exposure + "_FPS_" + str(
                self.video_fps)
            folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
            os.makedirs(folderStore, exist_ok=True)
            connected_image = folderStore + self.joinPathChar + fold_name + ".avi"

            # capture the image and save it on the save path
            os.makedirs(folderStore, exist_ok=True)

            # print(self.eV)
            video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.video_fps,
                                    (sv.width, sv.height))

            for i in range(len(reg_vid)):
                tempImg = Image.fromarray(reg_vid[i])
                # temp_img_plot = reg_vid_plot[i]

                # array = np.array(self.get_concat_h_blank(tempImg, temp_img_plot))
                array = np.array(tempImg)
                video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

            video.release()

            self.check_fps()

        elif self.res_check == 1 and self.current_auto_exposure == "None":

            self.check_fps()

            regular.main(self.scene[self.scene_index], self.video_fps, self.verSlider.get(), list, self.folders)

        elif self.res_check == 1 and self.current_auto_exposure == "Global" or self.current_auto_exposure == 'Local':

            self.check_fps()

            high_res_auto_ex_video.main(self.scene[self.scene_index], self.video_fps, self.eV,
                                        self.current_auto_exposure, self.folders)

    def get_concat_h_blank(self, im1, im2, color=(0, 0, 0)):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def check_fps(self):


        if self.validate_video_speed(self.save_video_fps.get()) is True:

            try:
                self.video_fps = int(self.save_video_fps.get())
                # print(set_speed)
            except ValueError:
                self.video_fps = 10  # set as default speed

    def check_num_grids(self):
        if len(self.rectangles) == 0:
            self.check_row_num_grids()
            self.check_col_num_grids()
        else:
            print(" Please clear rectangles before changing grid size")

    def check_row_num_grids(self):

        # print("row num girds is ", self.row_num_grids_.get())

        if self.validate_num_grids(self.row_num_grids_.get()) is True:

            try:
                self.row_num_grids = int(self.row_num_grids_.get())
                # print(set_speed)
            except ValueError:
                self.row_num_grids = 8  # set as default speed
        else:
            self.row_num_grids = 8

    def check_col_num_grids(self):

        # print("col num girds is ", self.col_num_grids_.get())

        if self.validate_num_grids(self.col_num_grids_.get()) is True:

            try:
                self.col_num_grids = int(self.col_num_grids_.get())
                # print(set_speed)
            except ValueError:
                self.col_num_grids = 8  # set as default speed
        else:
            self.col_num_grids = 8

    def pauseRun(self):
        self.play = False

    def runVideo(self):
        self.play = True
        self.playVideo()

    def setValues(self, dummy=False):
        self.runVideo()
        # self.play = True
        # self.playVideo()
        # time.sleep(1)
        # print(scene_name)
        self.check_num_grids()
        if self.scene[self.scene_index] != self.defScene.get():
            self.clear_rects()
            self.scene_index = self.scene.index(self.defScene.get())
            # input_ims = 'Image_Arrays_exposure/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
            self.setAutoExposure()

            # exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"], r_percent=self.exposureParams['r_percent'], g_percent=self.exposureParams['g_percent'],
            #                                     col_num_grids=self.exposureParams['col_num_grids'], row_num_grids=self.exposureParams['row_num_grids'], low_threshold=self.exposureParams['low_threshold'], start_index=self.exposureParams['start_index'],
            #                                     high_threshold=self.exposureParams['high_threshold'], high_rate=self.exposureParams['high_rate'])
            # self.eV,weighted_means,hists,hists_before_ds_outlier = exposures.pipeline()

            # self.img_mean_list = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.defScene.get() + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2 ** self.bit_depth - 1)

            self.img_mertens = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[
                    self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')

            self.img_raw = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                    self.scene_index] + '_show_dng_imgs' + '.npy')
            if self.scene_index < 20:
                self.img_all = np.load(os.path.join(os.path.dirname(__file__),
                                                    'Image_Arrays') + self.joinPathChar + self.defScene.get() + '_imgs_' + str(
                    self.downscale_ratio) + '.npy')
            else:
                self.img_all = self.img_raw

            self.resetValues()

    def setAutoExposure(self, dummy=False):
        self.current_auto_exposure = self.defAutoExposure.get()
        self.scene_index = self.scene.index(self.defScene.get())
        if self.stack_size[self.scene_index] == 40:
            input_ims = 'Image_Arrays_exposure_new/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        else:
            input_ims = 'Image_Arrays_exposure/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        self.check_num_grids()
        self.exposureParams = {"downsample_rate": 1 / 25, 'r_percent': 0.25, 'g_percent': 0.5,
                               'col_num_grids': self.col_num_grids, 'row_num_grids': self.row_num_grids,
                               'low_threshold': self.low_threshold.get(), 'start_index': int(self.start_index.get()),
                               'high_threshold': self.high_threshold.get(), 'high_rate': float(self.high_rate.get()),
                               'stepsize': self.stepsize_limit.get(),
                               "number_of_previous_frames": self.number_of_previous_frames.get(),
                               "global_rate":self.local_interested_global_area_percentage.get(),
                               "target_intensity":self.target_intensity.get()}
        if (self.current_auto_exposure == "Global"):
            self.clear_rects()
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
            self.clear_rects()
            name1 = self.scene[self.scene_index]+"_salient_maps_rbd.npy"
            name2 = self.scene[self.scene_index] + "_salient_maps_mbd.npy"

            #print(name)
            #salient_map_rbd = np.load("saliency_maps/"+name1)
            salient_map_mbd = np.load("saliency_maps/"+name2)
            #salient_map = (salient_map_mbd +salient_map_rbd)/2
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

            self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_with_salient_map(salient_map)

        elif (self.current_auto_exposure == "Entropy"):
            self.clear_rects()
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
            self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.entropy_pipeline()

        elif (self.current_auto_exposure == "Max Gradient srgb"):
            self.clear_rects()
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
            self.clear_rects()
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


        elif (self.current_auto_exposure == "Local"):
            self.clear_rects_local_wo_grids()
            consider_outliers = bool(self.local_consider_outliers_check)

            list_local = local_interested_grids_generater(self.row_num_grids, self.col_num_grids, self.rectangles)

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
                                                    'number_of_previous_frames']
                                                )
            self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline()
        elif (self.current_auto_exposure == "Local without grids"):
            self.clear_rects_local()
            list_local = self.list_local_without_grids()

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
            #exposures.gradient_exposure()
            self.eV,  self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_local_without_grids()

        elif (self.current_auto_exposure == "Local on moving objects"):
            self.clear_rects_local()
            list_local = self.list_local_without_grids_moving_objects()

            filenamePickle = os.path.join("local_pickle", f"Scene{self.scene_index}.pkl")
            with open(filenamePickle, 'wb') as f:
                pkl.dump(list_local, f)

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

    def list_local_without_grids(self):
        list_ = []
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for (y_start, x_strat, y_end, x_end) in self.rects_without_grids:
            list_.append([y_start / h, x_strat / w, y_end / h, x_end / w])
        return list_

    # since the list is small 100 * n * 4, n is usually < 3, and the n varies, list is used rather than numpy array
    def list_local_without_grids_moving_objects(self):
        list_ = []
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        number_of_frames = self.frame_num[self.scene_index]
        keys = self.rects_without_grids_moving_objests.keys()
        keys = sorted(keys)
        if len(keys) > 0:
            list_coordi_temp = self.rects_without_grids_moving_objests[keys[0]]
            # list_coordi_temp = []
            # for id in list_rectid_temp:
            #     coordi = self.canvas.coords(id)
            #     list_coordi_temp.append([coordi[1]/h,coordi[0]/w,coordi[3]/h,coordi[2]/w])
            for i in range(keys[0] + 1):
                list_.append(list_coordi_temp.copy())
            for i in range(1, len(keys)):
                list_pre = list_coordi_temp.copy()
                # list_coordi_temp = []
                list_coordi_temp = self.rects_without_grids_moving_objests[keys[i]]
                # for id in list_rectid_temp:
                #     coordi = self.canvas.coords(id)
                #     list_coordi_temp.append([coordi[1]/h, coordi[0]/w, coordi[3]/h, coordi[2]/w])
                gap = keys[i] - keys[i - 1]
                for j in range(1, gap):
                    # assume the number of rects are the same. if not, follow the less one, and assume the first "size" of rects are the cooresponding ones
                    size = min(len(list_pre), len(list_coordi_temp))
                    list_coordi_temp_gap = []
                    for k in range(size):
                        a1, b1, c1, d1 = list_pre[k]
                        a2, b2, c2, d2 = list_coordi_temp[k]
                        a = a1 + (a2 - a1) * j / gap
                        b = b1 + (b2 - b1) * j / gap
                        c = c1 + (c2 - c1) * j / gap
                        d = d1 + (d2 - d1) * j / gap
                        list_coordi_temp_gap.append([a, b, c, d])
                    list_.append(list_coordi_temp_gap.copy())
                list_.append(list_coordi_temp.copy())
            for i in range(keys[-1] + 1, self.frame_num[self.scene_index]):
                list_.append(list_coordi_temp.copy())
                # for (y_start,x_strat,y_end,x_end) in self.rects_without_grids_moving_objests[i]:
                #     list__.append([y_start/h,x_strat/w,y_end/h,x_end/w])
                # list_.append(list__)
        self.the_moving_area_list = list_.copy()

        print("MOVING AREA LIST", self.the_moving_area_list)
        return list_

    def playVideo(self):
        # global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play, video_speed, useMertens

        #

        if self.validate_video_speed(self.video_speed.get()) is True:

            try:
                set_speed = int(self.video_speed.get())
                # print(set_speed)
            except ValueError:
                set_speed = 360  # set as default speed

        # print('screen index is ', scene_index)

        if (self.horSlider.get() < (self.frame_num[self.scene_index] - 1) and self.play):
            self.horSlider.set(self.horSlider.get() + 1)
            # print("HELLO", horSlider.get())

            root.after(set_speed, self.playVideo)

        if (self.play is False):
            print("VIDEO PAUSED")

    def validate_video_speed(self, speed):
        # print("text is ", video_speed)
        try:
            if int(speed):
                return True
            else:
                return False
        except ValueError:
            return True

    def validate_num_grids(self, num):
        try:
            if int(num) and 1 < int(num) < 31:
                return True
            else:
                ("please enter an integer between 2 and 30")
                return False
        except ValueError:
            print("please enter an integer between 2 and 30")
            return False

    def resetValues(self):
        # global verSlider, horSlider, photo, img, scene_index, play, useMertens
        # if self.current_auto_exposure == "Local" or self.current_auto_exposure == "Local without grids":
        self.setAutoExposure()
        # self.useMertens = False
        # print("Reset")
        self.play = False
        # verSlider.config(to=stack_size[scene_index]-1)
        self.horSlider.config(to=self.frame_num[self.scene_index] - 1)
        # verSlider.set(0),
        self.horSlider.set(0)

        # self.imagePrevlabel.configure(image=photo)
        print("reset!")
        self.updatePlot()

    def updatePlot(self):
        # global verSlider, horSlider, photo, photo_2, stack_size, img_all, img, img_mean_list, scene_index, fig
        stack_size = self.stack_size[self.scene_index]
        if self.check == True:
            self.temp_img_ind = int(self.horSlider.get()) * stack_size + int(self.verSlider.get())
        else:
            pass

        self.check == True
        # Image mean plot
        if len(self.hists) != 0:

            first_ind = self.temp_img_ind // stack_size
            send_ind = self.temp_img_ind % stack_size
            count1 = self.hists[first_ind][send_ind]
            # print("current srgb hist check")
            # print(self.show_srgb_hist_check)
            if self.show_srgb_hist_check == 1:
                # print("here")
                count2, self.srgb_mean = self.show_srgb_hist()
                # print(count2)
            else:
                count2 = self.hists_before_ds_outlier[first_ind][send_ind]

            curr_frame_mean_list = self.weighted_means[first_ind]
            ind = send_ind
            val = curr_frame_mean_list[send_ind]
            ind2 = self.eV[self.horSlider.get()]
            print("IND2", ind2)
            val2 = curr_frame_mean_list[ind2]
            count3 = self.hists[first_ind][ind2]

        else:
            count1 = np.zeros(self.num_bins + 1)
            count2 = np.zeros(self.num_bins + 1)
            count3 = np.zeros(self.num_bins + 1)
            curr_frame_mean_list = np.zeros(stack_size)
            ind = 0
            val = 0
            ind2 = 0
            val2 = 0
        # self.hist_plot_unvisible()
        # self.image_mean_plot(stack_size=stack_size,curr_frame_mean_list=curr_frame_mean_list,ind=ind,val=val)
        # self.hist_plot(count1=count1, count2=count2)
        self.hist_plot_three(count1=count1, count2=count2, count3=count3, stack_size=stack_size,
                             curr_frame_mean_list=curr_frame_mean_list, ind=ind, val=val, ind2=ind2, val2=val2)
        if self.current_auto_exposure == "Local on moving objects":
            self.draw_interested_moving_areas_per_frame()

    def hist_laxis(self, data, n_bins,
                   range_limits):  # https://stackoverflow.com/questions/44152436/calculate-histograms-along-axis
        # Setup bins and determine the bin location for each element for the bins
        R = range_limits
        N = data.shape[-1]
        bins = np.linspace(R[0], R[1], n_bins + 1)
        data2D = data.reshape(-1, N)
        idx = np.searchsorted(bins, data2D, 'right') - 1

        # Some elements would be off limits, so get a mask for those
        bad_mask = (idx == -1) | (idx == n_bins)

        # We need to use bincount to get bin based counts. To have unique IDs for
        # each row and not get confused by the ones from other rows, we need to
        # offset each row by a scale (using row length for this).
        scaled_idx = n_bins * np.arange(data2D.shape[0])[:, None] + idx

        # Set the bad ones to be last possible index+1 : n_bins*data2D.shape[0]
        limit = n_bins * data2D.shape[0]
        scaled_idx[bad_mask] = limit

        # Get the counts and reshape to multi-dim
        counts = np.bincount(scaled_idx.ravel(), minlength=limit + 1)[:-1]
        counts.shape = data.shape[:-1] + (n_bins,)
        return counts

    def show_srgb_hist(self):  # assuming the channel order is RGB
        current_rgb_img = deepcopy(self.img_raw[self.horSlider.get()][self.verSlider.get()])
        current_rgb_img = current_rgb_img / (2**8 - 1)
        current_rgb_img[:, :, 0] = current_rgb_img[:, :, 0] * 0.2126
        current_rgb_img[:, :, 1] = current_rgb_img[:, :, 1] * 0.7152
        current_rgb_img[:, :, 2] = current_rgb_img[:, :, 2] * 0.0722
        current_rgb_img_ = np.sum(current_rgb_img, axis=2)

        if self.current_auto_exposure == "Local on moving objects" and len(self.the_moving_area_list) > 0:
            interested_boundaries = self.the_moving_area_list[self.horSlider.get()]
            temp_img = np.ones(current_rgb_img_.shape)*(-0.01)
            h,w = current_rgb_img_.shape
            #w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            for coord in interested_boundaries:
                w_start = int(coord[1]*w)
                h_start = int(coord[0]*h)
                w_end = min(int(coord[3]*w)+1,w+1)
                h_end = min(int(coord[2]*h)+1,h+1)
                # print("********")
                # print(w_start)
                # print(w_end)
                # print(h_start)
                # print(h_end)
                # print(w)
                # print(h)
                # print("--------------")
                temp_img[h_start:h_end,w_start:w_end] = current_rgb_img_[h_start:h_end,w_start:w_end]
            temp_img = temp_img.flatten()
            srgb_hist,dropped = self.get_hists(temp_img)
            mean = self.get_means(dropped, temp_img)
        else:
            mean = np.mean(current_rgb_img_)
            current_rgb_img_ = current_rgb_img_.flatten()
            srgb_hist = self.hist_laxis(current_rgb_img_, self.num_bins+1,
                                        (0, 1.01))
        return srgb_hist,mean

    def get_hists(self, flatten_weighted_ims):
        scene_hists_include_drooped_counts = self.hist_laxis(flatten_weighted_ims, self.num_bins + 2, (
            -0.01, 1.01))  # one extra bin is used to count the number of -0.01
        num_dropped_pixels = scene_hists_include_drooped_counts[0]
        scene_hists = scene_hists_include_drooped_counts[1:]
        return scene_hists, num_dropped_pixels

    def get_means(self, num_dropped_pixels, flatten_weighted_ims):
        weighted_all_means = np.mean(flatten_weighted_ims)
        if num_dropped_pixels == 0:
            return weighted_all_means
        c = len(flatten_weighted_ims)
        mean = (c*weighted_all_means + 0.01*num_dropped_pixels)/(c-num_dropped_pixels)
        return mean

    def clear_rects(self):
        self.clear_rects_local()
        self.clear_rects_local_wo_grids()
        self.clear_moving_rects()
        self.the_moving_area_list = []
        if self.making_a_serious_of_videos == 0:
            self.rects_without_grids_moving_objests = {}
        print("clear lenth of moving areas")
        print(len(self.the_moving_area_list))
        # self.rectangles = []
        # self.moving_rectids = []
        # self.the_scrolling_rect = []


    def clear_moving_rects(self):
        for rect in self.moving_rectids:
            self.canvas.delete(rect)
        self.moving_rectids = []
        if self.making_a_serious_of_videos == 0:
            self.rects_without_grids_moving_objests = {}

        #self.the_moving_area_list = []

    def clear_rects_local(self):
        self.rectangles = []
        for rect in self.current_rects:
            self.canvas.delete(rect)
        self.current_rects = []

    def clear_rects_local_wo_grids(self):
        self.rects_without_grids = []
        for rect in self.current_rects_wo_grids:
            self.canvas.delete(rect)
        self.current_rects_wo_grids = []

    def canvas_click(self, event):
        col, row = event.x, event.y

        # self.clear_rects()

        if self.current_auto_exposure == "Local":
            self.check_num_grids()
            self.colGridSelect = int(col * self.col_num_grids / self.photo.width())
            self.rowGridSelect = int(row * self.row_num_grids / self.photo.height())
            rect = [self.rowGridSelect, self.colGridSelect]
            self.rectangles.append(rect)  # making this array to allow us to be flexible in the future
            self.current_rects.append(self.draw_rectangle(rect[0], rect[1], "green"))
            self.setAutoExposure()

    def draw_rectangle(self, row, col, color):
        ww = self.photo.width()
        hh = self.photo.height()
        topx = col * (ww // self.col_num_grids)
        if col == self.col_num_grids - 1:
            botx = ww - 1
        else:
            botx = (col + 1) * (ww // self.col_num_grids)

        topy = row * (hh // self.row_num_grids)
        if row == self.row_num_grids - 1:
            boty = hh - 1
        else:
            boty = (row + 1) * (hh // self.row_num_grids)
        # print(topx, topy, botx, boty)
        rect = self.canvas.create_rectangle(topx, topy, botx, boty, fill='', outline=color)
        return rect

    def local_wo_grids(self, event):
        self.on_button_press(event)
        self.on_move_press(event)
        self.on_button_release(event)

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.current_auto_exposure == "Local without grids":
            # save mouse drag start position
            # create rectangle if not yet exist
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')
            self.current_rects_wo_grids.append(self.rect)
        if self.current_auto_exposure == "Local on moving objects":
            self.curX = self.start_x
            self.curY = self.start_y

            for i, r in enumerate(self.moving_rectids):

                r_start_x, r_start_y, r_end_x, r_end_y = self.canvas.coords(r)
                if r_start_x <= self.start_x <= r_end_x and r_start_y <= self.start_y <= r_end_y:
                    self.the_moving_rect = r
                    self.rect_ind = i
                    break
            # create rectangle if not yet exist
            if not self.the_moving_rect:
                rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='indigo')
                self.moving_rectids.append(rect)

    def on_move_press(self, event):
        if self.current_auto_exposure == "Local without grids":
            curX = self.canvas.canvasx(event.x)
            curY = self.canvas.canvasy(event.y)

            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            # print('w: '+str(w))
            # print('h: '+str(h))
            # print(event.x)
            # print(event.y)

            # expand rectangle as you drag the mouse

            self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
            self.curX = curX
            self.curY = curY
            # print("curx: "+ str(curX))
            # print("cury: "+ str(curY))
        if self.current_auto_exposure == "Local on moving objects":
            curX = self.canvas.canvasx(event.x)
            curY = self.canvas.canvasy(event.y)

            # expand rectangle as you drag the mouse
            if self.the_moving_rect == None:
                self.canvas.coords(self.moving_rectids[-1], self.start_x, self.start_y, curX, curY)
            else:
                self.x_offset = curX - self.curX
                self.y_offset = curY - self.curY
                # old_coordinate = self.canvas.coords(self.the_moving_rect)
                # x = old_coordinate[1] + self.x_offset
                # y = old_coordinate[0] + self.y_offset
                self.canvas.move(self.the_moving_rect, self.x_offset, self.y_offset)
            self.curX = curX
            self.curY = curY

    def on_button_release(self, event):
        print("rect: " + str(self.rect))
        print("start_x: " + str(self.start_x))
        print("start_y: " + str(self.start_y))
        print("cur_x: " + str(self.curX))
        print("cur_y: " + str(self.curY))
        if self.current_auto_exposure == "Local without grids":
            self.rects_without_grids.append([self.start_y, self.start_x, self.curY, self.curX])
            print(self.rects_without_grids)
        if self.current_auto_exposure == "Local":
            self.check_num_grids()
            self.colGridSelect = int(self.start_x * self.col_num_grids / self.photo.width())
            self.rowGridSelect = int(self.start_y * self.row_num_grids / self.photo.height())
            rect = [self.rowGridSelect, self.colGridSelect]
            self.rectangles.append(rect)  # making this array to allow us to be flexible in the future
            self.current_rects.append(self.draw_rectangle(rect[0], rect[1], "green"))
            self.setAutoExposure()
        if self.current_auto_exposure == "Local on moving objects":
            # print("rect: " + str(self.moving_rectids[-1]))
            # print("start_x: " + str(self.start_x))
            # print("start_y: " + str(self.start_y))
            # print("cur_x: " + str(self.curX))
            # print("cur_y: " + str(self.curY))
            if self.the_moving_rect != None:
                # self.x_offset = self.curX - self.start_x
                # self.y_offset = self.curY - self.start_y
                # self.canvas.move(self.the_moving_rect,self.x_offset,self.y_offset)
                # old_coordinate = self.canvas.coords(self.the_moving_rect)
                # print(old_coordinate)
                # self.rects[self.rect_ind] = [old_coordinate[1], old_coordinate[0], old_coordinate[3], old_coordinate[2]]
                self.the_moving_rect = None
                self.x_offset = 0
                self.y_offset = 0
                self.rect_ind = None

    def right_click(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.curX = self.start_x
        self.curY = self.start_y

        for i, r in enumerate(self.moving_rectids):

            r_start_x, r_start_y, r_end_x, r_end_y = self.canvas.coords(r)
            # r_start_y ,r_start_x,r_end_y,r_end_x   = self.rects[i]
            if r_start_x <= self.start_x <= r_end_x and r_start_y <= self.start_y <= r_end_y:
                self.the_scrolling_rect = r
                # self.the_rect_ind = i
                break

    def zoomerP(self, event):
        if self.the_scrolling_rect:
            old_coordinate = self.canvas.coords(self.the_scrolling_rect)
            factor = 1.1
            self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * 0.9, old_coordinate[1] * 0.9,
                               old_coordinate[2] * 1.1, old_coordinate[3] * 1.1)
            # self.canvas.configure(scrollregion = self.canvas.bbox("all"))

    def zoomerM(self, event):
        print(self.the_scrolling_rect)
        if self.the_scrolling_rect:
            old_coordinate = self.canvas.coords(self.the_scrolling_rect)
            print(old_coordinate)
            factor = 0.9
            self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * 1.1, old_coordinate[1] * 1.1,
                               old_coordinate[2] * 0.9, old_coordinate[3] * 0.9)

        # self.canvas.coords(self.the_moving_rect, event.x, event.y, 0.9, 0.9)
        # self.canvas.configure(scrollregion = self.canvas.bbox("all"))

    def zoomer(self, event):
        print("in zoomer")
        if self.the_scrolling_rect:
            print("here")
            if (event.delta > 0):
                old_coordinate = self.canvas.coords(self.the_scrolling_rect)
                factor = 1.1
                self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * 0.9, old_coordinate[1] * 0.9,
                                   old_coordinate[2] * 1.1, old_coordinate[3] * 1.1)
            elif (event.delta < 0):
                old_coordinate = self.canvas.coords(self.the_scrolling_rect)
                print(old_coordinate)
                factor = 0.9
                self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * 1.1, old_coordinate[1] * 1.1,
                                   old_coordinate[2] * 0.9, old_coordinate[3] * 0.9)

    def updateSlider(self, scale_value):
        if ((self.current_auto_exposure != "None") and (len(self.eV) > 0)):
            print(len(self.eV))
            self.verSlider.set(self.eV[self.horSlider.get()])
            temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + self.eV[self.horSlider.get()]
        else:
            temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + int(self.verSlider.get())

        self.updateHorSlider(scale_value, temp_img_ind)
        # autoExposureMode = True
        #
        # # global verSlider, horSlider, photo, photo_2, imagePrevlabel, imagePrevlabel_2, img_all, img, img_mean_list, scene_index, fig, useMertens, mertensVideo
        # #
        # print(self.useRawIms)
        # if self.useMertens:
        #     # img = self.mertensVideo[self.horSlider.get()]
        #     img = self.img_mertens[self.horSlider.get()]
        # elif self.useRawIms:
        #     #print(self.verSlider.get())
        #     img = self.img_raw[self.horSlider.get()][self.verSlider.get()]
        # else:
        #     img = deepcopy(self.img_all[temp_img_ind])
        #
        # # self.photo = ImageTk.PhotoImage(Image.fromarray(self.img))
        # # self.imagePrevlabel = tk.Label(root, image=self.photo)
        # # self.imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)
        #
        # tempImg = Image.fromarray(img).resize((self.canvas.winfo_width(),self.canvas.winfo_height()))
        #
        #
        #
        # self.photo = ImageTk.PhotoImage(tempImg,width=self.canvas.winfo_width(),height=self.canvas.winfo_height())
        #  #= self.photo  # Keep reference in case this code is put into a function.
        #
        # self.canvas.itemconfig(self.canvas_img, image=self.photo)
        #
        # self.canvas.tag_lower(self.canvas_img)
        #
        # # Keep reference in case this code is put into a function.
        # self.updatePlot()

    def updateHorSlider(self, scale_value, temp_img_ind):

        autoExposureMode = True

        # global verSlider, horSlider, photo, photo_2, imagePrevlabel, imagePrevlabel_2, img_all, img, img_mean_list, scene_index, fig, useMertens, mertensVideo
        #
        if self.useMertens:
            # img = self.mertensVideo[self.horSlider.get()]
            img = self.img_mertens[self.horSlider.get()]
        elif self.useRawIms:
            # print(self.verSlider.get())
            img = self.img_raw[self.horSlider.get()][self.verSlider.get()]
        else:
            img = deepcopy(self.img_all[temp_img_ind])

        # self.photo = ImageTk.PhotoImage(Image.fromarray(self.img))
        # self.imagePrevlabel = tk.Label(root, image=self.photo)
        # self.imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)

        tempImg = Image.fromarray(img).resize((self.canvas.winfo_width(), self.canvas.winfo_height()))

        self.photo = ImageTk.PhotoImage(tempImg, width=self.canvas.winfo_width(), height=self.canvas.winfo_height())
        # = self.photo  # Keep reference in case this code is put into a function.

        self.canvas.itemconfig(self.canvas_img, image=self.photo)

        self.canvas.tag_lower(self.canvas_img)

        # Keep reference in case this code is put into a function.
        self.updatePlot()

    def make_global_videos(self):
        if self.current_auto_exposure != "Global":
            return
        line_labels = [# "grid size: 8*8; outlier boundary: 0 & 1; previous # of frames: 1; step limitation: 40",
                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 1; step limitation: 0",
                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 1; step limitation: 40",
                       # "grid size: 8*8; outlier boundary: 0.1 & 0.8; previous # of frames: 1; step limitation: 40",
                       # "grid size: 20*20; outlier boundary: 0.1 & 0.8; previous # of frames: 1; step limitation: 40",
                       # "grid size: 20*20; outlier boundary: 0 & 0.9; previous # of frames: 1; step limitation: 40",
                       # "grid size: 20*20; outlier boundary: 0 & 1; previous # of frames: 1; step limitation: 40",
                       # "grid size: 20*20; outlier boundary: 0 & 1; previous # of frames: 10; step limitation: 1",
                       # "grid size: 20*20; outlier boundary: 0 & 1; previous # of frames: 5; step limitation: 3",
                       # "grid size: 20*20; outlier boundary: 0 & 0.9; previous # of frames: 5; step limitation: 3",
                       # "grid size: 20*20; outlier boundary: 0.1 & 0.8; previous # of frames: 5; step limitation: 3",
                       # "grid size: 20*20; outlier boundary: 0.1 & 0.8; previous # of frames: 10; step limitation: 1",
                       # "grid size: 20*20; outlier boundary: 0 & 0.9; previous # of frames: 10; step limitation: 1",
                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 10; step limitation: 1",
                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 5; step limitation: 3",
                       # "grid size: 8*8; outlier boundary: 0.1 & 0.8; previous # of frames: 5; step limitation: 3",
                       # "grid size: 8*8; outlier boundary: 0.1 & 0.8; previous # of frames: 10; step limitation: 1",
                       # "grid size: 8*8; outlier boundary: 0 & 1; previous # of frames: 10; step limitation: 1",
                       # "grid size: 8*8; outlier boundary: 0 & 1; previous # of frames: 5; step limitation: 3"
                       ]
        line_values = []
        self.scene_index = self.scene.index(self.defScene.get())
        if self.stack_size[self.scene_index] == 40:
            input_ims = 'Image_Arrays_exposure_new/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        else:
            return


        col_num_grids = 8
        row_num_grids = 8
        low_threshold = 0
        start_index = 0
        high_threshold = 0.9
        high_rate = 0
        stepsize_limit = 0
        number_of_previous_frames = 1
        downsample_rate = 1 / 25
        r_percent = 0.25
        g_percent = 0.5
        target_intensity = self.target_intensity.get()
        # 8 8 0 1 100 1
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())

        low_threshold = 0
        high_threshold = 0.9
        # 8 8 0 0.9 100 1
        self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        stepsize_limit = 40

        self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        stepsize_limit = 1
        number_of_previous_frames = 10

        self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        stepsize_limit = 3
        number_of_previous_frames = 5

        self.make_global_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate,
                                       col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        # low_threshold = 0.1
        # high_threshold = 0.8
        # # 8 8 0.1 0.8 100 1
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())

        # col_num_grids = 20
        # row_num_grids = 20
        # # 20 20 0.1 0.8 100 1
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())

        # low_threshold = 0.05
        # high_threshold = 0.9
        # # 20 20 0.05 0.9 100 1
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())

        # low_threshold = 0
        # high_threshold = 1
        # # 20 20 0 1 40 1
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 1
        # number_of_previous_frames = 10
        # # 20 20 0 1 1 10
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 3
        # number_of_previous_frames = 5
        # # 20 20 0 1 3 5
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # low_threshold = 0.05
        # high_threshold = 0.9
        # # 20 20 0.05 0.9 3 5
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # low_threshold = 0.1
        # high_threshold = 0.8
        # # 20 20 0.1 0.8 3 5
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 1
        # number_of_previous_frames = 10
        # # 20 20 0.1 0.8 1 10
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # low_threshold = 0.05
        # high_threshold = 0.9
        # # 20 20 0.05 0.9 1 10
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # col_num_grids = 8
        # row_num_grids = 8
        # # 8 8 0.05 0.9 1 10
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 3
        # number_of_previous_frames = 5
        # # 8 8 0.05 0.9 3 5
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # low_threshold = 0.1
        # high_threshold = 0.8
        # # 8 8 0.1 0.8 3 5
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 1
        # number_of_previous_frames = 10
        # # 8 8 0.1 0.8 1 10
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # low_threshold = 0
        # high_threshold = 1
        # # 8 8 0 1 1 10
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 3
        # number_of_previous_frames = 5
        # # 8 8 0 1 3 5
        # self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                                number_of_previous_frames)
        # line_values.append(self.eV.copy())
        #df = pd.DataFrame({"Settings": line_labels,'Shutter Speed': ylable})
        with open(str(self.scene[self.scene_index])+'_global.csv', 'w') as file:
            file.write("Settings,frame_number,Shutter_Speed")
            file.write('\n')
            for i,setting in enumerate(line_labels):
                for j, shutterspeedind in enumerate(line_values[i]):
                    file.write(setting + "," + str(j) + "," + str(self.NEW_SCALES[shutterspeedind])+"\n")



    def make_local_videos(self):
        if self.current_auto_exposure != "Local without grids":
            return
        self.scene_index = self.scene.index(self.defScene.get())
        if self.stack_size[self.scene_index] == 40:
            input_ims = 'Image_Arrays_exposure_new/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        else:
            return
        line_labels = ["outlier boundary: 0 & 1; previous # of frames: 1; step limitation: 40",
                       "outlier boundary: 0 & 1; previous # of frames: 10; step limitation: 1",
                       "outlier boundary: 0 & 1; previous # of frames: 5; step limitation: 3",
                       "outlier boundary: 0.05 & 0.9; previous # of frames: 5; step limitation: 3",
                       "outlier boundary: 0.05 & 0.9; previous # of frames: 10; step limitation: 1",
                       "outlier boundary: 0.05 & 0.9; previous # of frames: 1; step limitation: 40"
                       ]
        line_values = []

        col_num_grids = 8
        row_num_grids = 8
        low_threshold = 0
        start_index = 15
        high_threshold = 1
        high_rate = 0.2
        stepsize_limit = 5
        number_of_previous_frames = 1
        downsample_rate = 1 / 25
        r_percent = 0.25
        g_percent = 0.5
        target_intensity = self.target_intensity.get()
        global_rate = self.local_interested_global_area_percentage.get()
        #  0 1 100 1
        self.make_local_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 1
        number_of_previous_frames = 10
        #  0 1 1 10
        self.make_local_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())
        print(line_values)

        stepsize_limit = 3
        number_of_previous_frames = 5
        # 0 1 3 5
        self.make_local_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        low_threshold = 0
        high_threshold = 0.9
        # 0 0.9 3 5
        self.make_local_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 1
        number_of_previous_frames = 10
        # 0.05 0.9 1 10
        self.make_local_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 40
        number_of_previous_frames = 1
        # 0.05 0.9 100 1
        self.make_local_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames, global_rate)
        line_values.append(list(self.eV.copy()))
        #df = pd.DataFrame({"Settings": line_labels,'Shutter Speed': ylable})
        with open(str(self.scene[self.scene_index])+'_local_'+self.local_interested_name.get()+'.csv', 'w') as file:
            file.write("Settings,frame_number,Shutter_Speed")
            file.write('\n')
            for i,setting in enumerate(line_labels):
                for j, shutterspeedind in enumerate(line_values[i]):
                    # print(i)
                    # print(j)
                    # print(shutterspeedind)
                    file.write(setting + "," + str(j) + "," + str(self.NEW_SCALES[shutterspeedind])+"\n")


    def make_moving_object_videos(self):
        if self.current_auto_exposure != "Local on moving objects":
            return
        self.making_a_serious_of_videos = 1
        self.scene_index = self.scene.index(self.defScene.get())
        if self.stack_size[self.scene_index] == 40:
            input_ims = 'Image_Arrays_exposure_new/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
        else:
            return
        line_labels = ["outlier boundary: 0 & 1; previous # of frames: 1; step limitation:0",
                       "outlier boundary: 0 & 1; previous # of frames: 1; step limitation:40",
                       "outlier boundary: 0 & 1; previous # of frames: 10; step limitation: 1",
                       "outlier boundary: 0 & 1; previous # of frames: 5; step limitation: 3",
                       "cropped; outlier boundary: 0 & 1; previous # of frames: 1; step limitation:0",
                       "cropped; outlier boundary: 0 & 1; previous # of frames: 1; step limitation:40",
                       "cropped; outlier boundary: 0 & 1; previous # of frames: 10; step limitation: 1",
                       "cropped; outlier boundary: 0 & 1; previous # of frames: 5; step limitation: 3",

                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 1; step limitation: 0",
                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 1; step limitation: 40",

                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 10; step limitation: 1",
                       "grid size: 8*8; outlier boundary: 0 & 0.95; previous # of frames: 5; step limitation: 3"

                       ]
        line_values = []

        col_num_grids = 8
        row_num_grids = 8
        low_threshold = 0
        start_index = 15
        high_threshold = 0.9
        high_rate = 0.2
        stepsize_limit = 0
        number_of_previous_frames = 1
        downsample_rate = 1 / 25
        r_percent = 0.25
        g_percent = 0.5
        target_intensity = self.target_intensity.get()
        global_rate = self.local_interested_global_area_percentage.get()


        #
        #   0 1 0 1
        self.make_moving_object_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate,
                                              col_num_grids, row_num_grids,
                                              low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                              number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        #
        stepsize_limit = 40
        #   0 1 40 1
        self.make_moving_object_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate,
                                              col_num_grids, row_num_grids,
                                              low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                              number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 1
        number_of_previous_frames = 10
        #  0 1 1 10
        self.make_moving_object_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate,
                                              col_num_grids, row_num_grids,
                                              low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                              number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 3
        number_of_previous_frames = 5
        #
        # 0 1 3 5
        self.make_moving_object_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate,
                                              col_num_grids, row_num_grids,
                                              low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                              number_of_previous_frames, global_rate)
        line_values.append(self.eV.copy())


        self.make_crop_video_flag = 1

        stepsize_limit = 0
        number_of_previous_frames = 1
        #  0 1 0 1
        self.make_moving_object_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames,global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 40
        #  0 1 40 1
        self.make_moving_object_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames,global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 1
        number_of_previous_frames = 10
        #  0 1 1 10
        self.make_moving_object_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames,global_rate)
        line_values.append(self.eV.copy())

        stepsize_limit = 3
        number_of_previous_frames = 5

        # 0 1 3 5
        self.make_moving_object_videos_helper(input_ims, target_intensity,r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                number_of_previous_frames,global_rate)
        line_values.append(self.eV.copy())
        #
        # low_threshold = 0.05
        # high_threshold = 0.9
        # # 0.05 0.9 3 5
        # self.make_moving_object_videos_helper(input_ims, target_intensity,r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                         low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                         number_of_previous_frames,global_rate)
        # line_values.append(self.eV.copy())
        #
        # stepsize_limit = 1
        # number_of_previous_frames = 10
        # # 0.05 0.9 1 10
        # self.make_moving_object_videos_helper(input_ims, target_intensity,r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                         low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                         number_of_previous_frames,global_rate)
        # line_values.append(self.eV.copy())
        #
        #
        # stepsize_limit = 40
        # number_of_previous_frames = 1
        # # 0.05 0.9 100 1
        # self.make_moving_object_videos_helper(input_ims, target_intensity,r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
        #                         low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
        #                         number_of_previous_frames,global_rate)
        #
        # line_values.append(list(self.eV.copy()))

        # "cropped; outlier boundary: 0 & 1; previous # of frames: 5; step limitation: 3",
        # "cropped; outlier boundary: 0 & 1; previous # of frames: 1; step limitation:0",
        # "cropped; outlier boundary: 0 & 1; previous # of frames: 1; step limitation:40",
        # "cropped; outlier boundary: 0 & 1; previous # of frames: 10; step limitation: 1",





        self.current_auto_exposure = "Global"
        self.defAutoExposure.set("Global")
        # cropped video on global selection
        col_num_grids = 8
        row_num_grids = 8
        low_threshold = 0
        start_index = 15
        high_threshold = 1
        high_rate = 0.2
        stepsize_limit = 0
        number_of_previous_frames = 1
        downsample_rate = 1 / 25
        r_percent = 0.25
        g_percent = 0.5
        target_intensity = self.target_intensity.get()

        low_threshold = 0
        high_threshold = 0.9
        # 8 8 0 0.9 100 1
        self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        stepsize_limit = 40

        self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        stepsize_limit = 1
        number_of_previous_frames = 10

        self.make_global_videos_helper(input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        stepsize_limit = 3
        number_of_previous_frames = 5

        self.make_global_videos_helper(input_ims, target_intensity, r_percent, g_percent, downsample_rate,
                                       col_num_grids, row_num_grids,
                                       low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                       number_of_previous_frames)
        line_values.append(self.eV.copy())

        self.make_crop_video_flag = 0
        self.temp_coords_for_video_producing = []
        self.making_a_serious_of_videos = 0


        #df = pd.DataFrame({"Settings": line_labels,'Shutter Speed': ylable})
        with open(str(self.scene[self.scene_index])+'_local_moving_'+self.local_interested_name.get()+'.csv', 'w') as file:
            file.write("Settings,frame_number,Shutter_Speed")
            file.write('\n')
            for i,setting in enumerate(line_labels):
                for j, shutterspeedind in enumerate(line_values[i]):
                    # print(i)
                    # print(j)
                    # print(shutterspeedind)
                    file.write(setting + "," + str(j) + "," + str(self.NEW_SCALES[shutterspeedind])+"\n")

        self.current_auto_exposure = "Local on moving objects"
        self.defAutoExposure.set("Local on moving objects")


    def make_moving_object_videos_helper(self, input_ims,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                 low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                 number_of_previous_frames,local_interested_global_area_percentage):
        exposureparams = self.set_params(target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                         low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                         number_of_previous_frames,local_interested_global_area_percentage)
        # self.clear_rects()
        exposures = self.exposure_class_construction_moving_object(input_ims, exposureparams)
        # exposures = exposure_class.Exposure(params = self.exposureParams)

        self.eV,  self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_local_without_grids_moving_object()
        print(self.eV)
        self.export_video_2(col_num_grids, row_num_grids, low_threshold, start_index, high_threshold, high_rate,
                            stepsize_limit, number_of_previous_frames, target_intensity)

    def make_local_videos_helper(self, input_ims, target_intensity,r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                 low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                 number_of_previous_frames,local_interested_global_area_percentage):
        exposureparams = self.set_params(target_intensity,r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                         low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                         number_of_previous_frames,local_interested_global_area_percentage)
        # self.clear_rects()
        exposures = self.exposure_class_construction_local(input_ims, exposureparams)
        # exposures = exposure_class.Exposure(params = self.exposureParams)
        self.eV, self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline_local_without_grids()
        self.export_video_2(col_num_grids, row_num_grids, low_threshold, start_index, high_threshold, high_rate,
                            stepsize_limit, number_of_previous_frames, target_intensity)

    def make_global_videos_helper(self, input_ims, target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                  low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                  number_of_previous_frames):
        exposureparams = self.set_params(target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids,
                                         low_threshold, start_index, high_threshold, high_rate, stepsize_limit,
                                         number_of_previous_frames,0)
        self.clear_rects()
        exposures = self.exposure_class_construction(input_ims, exposureparams)
        # exposures = exposure_class.Exposure(params = self.exposureParams)
        self.eV,  self.eV_original, self.weighted_means, self.hists, self.hists_before_ds_outlier = exposures.pipeline()
        self.export_video_2(col_num_grids, row_num_grids, low_threshold, start_index, high_threshold, high_rate,
                            stepsize_limit, number_of_previous_frames, target_intensity)

    def set_params(self,target_intensity, r_percent, g_percent, downsample_rate, col_num_grids, row_num_grids, low_threshold, start_index,
                   high_threshold, high_rate, stepsize_limit, number_of_previous_frames,local_interested_global_area_percentage):
        exposureParams = {"downsample_rate": downsample_rate, 'r_percent': r_percent, 'g_percent': g_percent,
                          'col_num_grids': col_num_grids, 'row_num_grids': row_num_grids,
                          'low_threshold': low_threshold, 'start_index': start_index,
                          'high_threshold': high_threshold, 'high_rate': high_rate, 'stepsize': stepsize_limit,
                          "number_of_previous_frames": number_of_previous_frames,
                          "global_rate":local_interested_global_area_percentage,
                          "target_intensity":target_intensity}
        return exposureParams

    def exposure_class_construction_local(self, input_ims, exposureparams):
        self.clear_rects_local()
        list_local = self.list_local_without_grids()

        exposures = exposure_class.Exposure(input_ims, downsample_rate=exposureparams["downsample_rate"],
                                            r_percent=exposureparams['r_percent'],
                                            g_percent=exposureparams['g_percent'],
                                            col_num_grids=exposureparams['col_num_grids'],
                                            row_num_grids=exposureparams['row_num_grids'],
                                            low_threshold=exposureparams['low_threshold'],
                                            start_index=exposureparams['start_index'],
                                            high_threshold=exposureparams['high_threshold'],
                                            high_rate=exposureparams['high_rate'], local_indices=list_local,
                                            stepsize=exposureparams['stepsize'],
                                            number_of_previous_frames=exposureparams[
                                                'number_of_previous_frames'],
                                            target_intensity=exposureparams['target_intensity'],
                                            global_rate=exposureparams['global_rate']
                                            )
        return exposures

    def exposure_class_construction_moving_object(self, input_ims, exposureparams):
        self.clear_rects_local()
        list_local = self.list_local_without_grids_moving_objects()

        exposures = exposure_class.Exposure(input_ims, downsample_rate=exposureparams["downsample_rate"],
                                            r_percent=exposureparams['r_percent'],
                                            g_percent=exposureparams['g_percent'],
                                            col_num_grids=exposureparams['col_num_grids'],
                                            row_num_grids=exposureparams['row_num_grids'],
                                            low_threshold=exposureparams['low_threshold'],
                                            start_index=exposureparams['start_index'],
                                            high_threshold=exposureparams['high_threshold'],
                                            high_rate=exposureparams['high_rate'], local_indices=list_local,
                                            stepsize=exposureparams['stepsize'],
                                            number_of_previous_frames=exposureparams[
                                                'number_of_previous_frames'],
                                            target_intensity=exposureparams['target_intensity'],
                                            global_rate=exposureparams['global_rate']
                                            )
        return exposures
    def exposure_class_construction(self, input_ims, exposureparams):
        exposures = exposure_class.Exposure(input_ims, downsample_rate=exposureparams["downsample_rate"],
                                            r_percent=exposureparams['r_percent'],
                                            g_percent=exposureparams['g_percent'],
                                            col_num_grids=exposureparams['col_num_grids'],
                                            row_num_grids=exposureparams['row_num_grids'],
                                            low_threshold=exposureparams['low_threshold'],
                                            high_threshold=exposureparams['high_threshold'],
                                            high_rate=exposureparams['high_rate'],
                                            stepsize=exposureparams['stepsize'],
                                            number_of_previous_frames=exposureparams['number_of_previous_frames'],
                                            target_intensity=exposureparams['target_intensity'],
                                            start_index=exposureparams['start_index'])
        return exposures

    def export_video_2(self, col_num_grids, row_num_grids, low_threshold, start_index, high_threshold, high_rate,
                       stepsize_limit, number_of_previous_frames, target_intensity):

        reg_vid = []
        # reg_vid_plot = []
        # list = ['15', '8', '6', '4', '2', '1', '05', '1-4', '1-8', '1-15', '1-30', '1-60', '1-125', '1-250', '1-500']

        self.mertensVideo = []
        self.mertens_pic = []
        frame_size = self.frame_num[self.scene_index]
        if not len(self.eV) == frame_size:
            return

        for i in range(frame_size):
            self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + self.eV[i]
            self.check = False
            self.updatePlot()
            # reg_vid_plot.append(self.tempImg_2)

            # img = deepcopy(self.img_all[self.temp_img_ind])
            img = deepcopy(self.img_raw[i][self.eV[i]])
            # cropped video only works when the number of interested area is one
            # if self.make_crop_video_flag == 1 and len(self.the_moving_area_list) > 0 and len(
            #         self.the_moving_area_list[0]) == 1:
            if self.make_crop_video_flag == 1:
                if len(self.temp_coords_for_video_producing) < self.frame_num[self.scene_index]:
                    self.temp_coords_for_video_producing.append((self.the_moving_area_list[i][0]).copy())
                coord = self.temp_coords_for_video_producing[i]
                # temp_img = np.ones(current_rgb_img_.shape) * (-0.01)
                # h, w = current_rgb_img_.shape
                # w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
                # for coord in interested_boundaries:
                h, w, x = img.shape
                w_start = int(coord[1] * w)
                h_start = int(coord[0] * h)
                w_end = min(int(coord[3] * w) + 1, w + 1)
                h_end = min(int(coord[2] * h) + 1, h + 1)
                # print("********")
                # print(w_start)
                # print(w_end)
                # print(h_start)
                # print(h_end)
                # print(w)
                # print(h)
                # print("--------------")
                img = img[h_start:h_end, w_start:w_end, :]
            reg_vid.append(img)

        m1 = Image.fromarray(reg_vid[0])
        # m2 = reg_vid_plot[0]
        # sv = self.get_concat_h_blank(m1, m2)
        sv = m1

        self.check_fps()
        if self.current_auto_exposure == "Global":
            fold_name = self.scene[self.scene_index] + "_dng_pipeline_" + self.current_auto_exposure + "_" + str(col_num_grids) + "x" + str(row_num_grids) + "_low_threshold" + str(
                low_threshold) + "(" + str(start_index) + ")_high_threshold" + str(high_threshold) + "(" + str(
                high_rate) + ")_steplimit" + str(stepsize_limit) + "_PreFrames" + str(number_of_previous_frames) + "_target_mean_" + str(target_intensity)
        else:
            fold_name = self.scene[
                            self.scene_index] + "_dng_pipeline_local_" + self.local_interested_name.get() + '_steplimit' + str(stepsize_limit) + "_PreFrames" + str(number_of_previous_frames)+ "_target_mean_" + str(target_intensity)
        if self.make_crop_video_flag == 1:
            fold_name = fold_name + "_cropped"

        folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
        os.makedirs(folderStore, exist_ok=True)
        connected_image = folderStore + self.joinPathChar + self.scene[self.scene_index] + '/' + fold_name + ".avi"

        # capture the image and save it on the save path
        os.makedirs(folderStore, exist_ok=True)

        # print(self.eV)
        video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.video_fps,
                                (sv.width, sv.height))

        for i in range(len(reg_vid)):
            tempImg = Image.fromarray(reg_vid[i])
            # temp_img_plot = reg_vid_plot[i]

            # array = np.array(self.get_concat_h_blank(tempImg, temp_img_plot))
            array = np.array(tempImg)
            video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

        video.release()

        self.check_fps()


b = Browser(root)

root.mainloop()
