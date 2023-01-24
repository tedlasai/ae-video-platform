import constants
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


root = tk.Tk()
root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5
class Browser:

    def __init__(self, root):
        super().__init__()
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

        self.img_all = np.load(
            os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                self.scene_index] + '_show_dng_imgs' + '.npy')

        self.img = deepcopy(self.img_all[0,0])
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
        self.num_bins = 100
        self.hists = []
        self.hists_before_ds_outlier = []

        self.fig_2 = None
        self.fig = None
        self.fig_4 = None
        self.local_consider_outliers_check = 0
        self.show_srgb_hist_check = 0
        self.srgb_mean = 0
        # self.init_functions()
        # self.show_srgb_hist_check = self.show_srgb_hist_check_.get()


b = Browser(root)

root.mainloop()
