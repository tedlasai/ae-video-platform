import tkinter
import tkinter as tk
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

mp.rcParams.update({'axes.titlesize': 14, 'font.size': 11, 'font.family': 'arial'})

#Tkinter Window
root=tk.Tk()
root.geometry('1600x900'), root.title('Data Browser') #1900x1000+5+5

class Browser:

    def __init__(self, root):
        super().__init__()

        self.folders = "D:\Final"     #link to directory containing all the dataset image folders

        self.widgetFont = 'Arial'
        self.widgetFontSize = 12

        self.scene = ['Scene101', 'Scene102', 'Scene103', 'Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
                      'Scene7', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15', 'Scene16', 'Scene17', 'Scene18']
        self.frame_num = [90, 65, 15, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # number of frames per position
        self.stack_size = [12, 47, 28, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]  # number of shutter options per position

        self.scene = ['Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
                      'Scene7', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15',
                      'Scene16', 'Scene17', 'Scene18', 'Scene19', 'Scene20']
        self.frame_num = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                          100, 100, 100, 100]  # number of frames per position
        self.stack_size = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                           15, 15, 15]  # number of shutter options per position

        self.auto_exposures = ["None", "Global", "Local",'Local without grids']
        self.current_auto_exposure = "None"

        self.scene_index = 0
        self.mertensVideo = []
        self.bit_depth = 8
        self.downscale_ratio = 0.12
        self.check = True
        self.temp_img_ind = 0

        self.joinPathChar = "/"
        if (platform.system() == "Windows"):
            self.joinPathChar = "\\"

        self.imgSize = [int(4480 * self.downscale_ratio), int(6720 * self.downscale_ratio)]
        self.widthToScale = self.imgSize[1]
        self.widPercent = (self.widthToScale / float(self.imgSize[1]))
        self.heightToScale = int(float(self.imgSize[0]) * float(self.widPercent))



        self.img_all = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[self.scene_index] + '_imgs_' + str(self.downscale_ratio) + '.npy')
        self.img_mean_list = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[self.scene_index] + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2**self.bit_depth - 1)
        self.img_mertens = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')
        self.img_raw = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[self.scene_index] + '_show_dng_imgs' + '.npy')

        self.img = deepcopy(self.img_all[0])
        self.useMertens = False
        self.useRawIms = 0
        self.play = True
        self.video_speed = 50
        self.video_fps = 30

        self.res_check = 0
        self.hdr_mode_check = 0

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
        #self.canvas.bind('<Button-1>', self.canvas_click)
        self.canvas.bind('<ButtonPress-1>',self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_move_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        #some defaults
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
        self.init_functions()

    def init_functions(self):

        self.hdr_mean_button()
        self.hdr_median_button()
        self.hdr_mertens_button()
        self.hdr_abdullah_button()
        self.hdr_run_button()
        self.hdr_pause_button()
        self.hdr_reset_button()
        self.scene_select()
        self.playback_text_box()
        self.video_fps_text()
        self.horizontal_slider()
        self.vertical_slider()
        # self.image_mean_plot()
        # self.hist_plot()
        self.hist_plot_three()
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


    def hdr_mean_button(self):
        # HDR Button - Mean
        self.HdrMeanButton = tk.Button(root, text='HDR-Mean', fg='#ffffff', bg='#999999', activebackground='#454545',
                                  relief=tk.RAISED, width=16,padx=10, pady=5, font=(self.widgetFont, self.widgetFontSize), command=self.HdrMean)
        self.HdrMeanButton.grid(row=1, column=5, sticky=tk.E)  # initial row was 26, +1 increments for all other rows

    def hdr_median_button(self):
        # HDR Button - Median
        self.HdrMedianButton = tk.Button(root, text='HDR-Median', fg='#ffffff', bg='#999999', activebackground='#454545',
                                    relief=tk.RAISED, width=16, padx=10, pady=5,font=(self.widgetFont, self.widgetFontSize), command=self.HdrMedian)
        self.HdrMedianButton.grid(row=2, column=5, sticky=tk.E)

    def hdr_mertens_button(self):
        # HDR Button - Mertens
        self.HdrMertensButton = tk.Button(root, text='HDR-Mertens', fg='#ffffff', bg='#999999', activebackground='#454545',
                                     relief=tk.RAISED, width=16,padx=10, pady=5, font=(self.widgetFont, self.widgetFontSize), command=self.HdrMertens)
        self.HdrMertensButton.grid(row=3, column=5, sticky=tk.E)

    def show_Raw_Ims_check_box(self):
        self.useRawIms_ = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text='Show Raw Image', variable=self.useRawIms_, offvalue=0, onvalue=1,command=self.switch_raw)
        self.c1.grid(row=24, column=5)

    def hdr_abdullah_button(self):
        # HDR Button - Abdullah
        self.HdrAbdullahButton = tk.Button(root, text='HDR-Abdullah', fg='#ffffff', bg='#999999', activebackground='#454545',
                                      relief=tk.RAISED, width=16,padx=10, pady=5, font=(self.widgetFont, self.widgetFontSize),
                                      command=self.HdrAbdullah)
        self.HdrAbdullahButton.grid(row=4, column=5, sticky=tk.E)

    def hdr_run_button(self):
        # Run Button
        self.RunButton = tk.Button(root, text='Run', fg='#ffffff', bg='#999999', activebackground='#454545',
                              relief=tk.RAISED, width=16, padx=10, pady=5,font=(self.widgetFont, self.widgetFontSize), command=self.runVideo)
        self.RunButton.grid(row=5, column=5, sticky=tk.E)

    def hdr_pause_button(self):
        self.PauseButton = tk.Button(root, text='Pause', fg='#ffffff', bg='#999999', activebackground='#454545',
                                relief=tk.RAISED,padx=10, pady=5,
                                width=16, font=(self.widgetFont, self.widgetFontSize), command=self.pauseRun)
        self.PauseButton.grid(row=6, column=5, sticky=tk.E)

    def hdr_reset_button(self):
        # Reset Button
        self.RestButton = tk.Button(root, text='Reset', fg='#ffffff', bg='#999999', activebackground='#454545',
                               relief=tk.RAISED,padx=10, pady=5, width=16, font=(self.widgetFont, self.widgetFontSize), command=self.resetValues)
        self.RestButton.grid(row=7, column=5, sticky=tk.E)

    def regular_video_button(self):

        self.VideoButton = tk.Button(root, text='Video', fg='#ffffff', bg='#999999', activebackground='#454545',
                                relief=tk.RAISED,padx=10, pady=5,
                                width=16, font=(self.widgetFont, self.widgetFontSize), command=self.export_video)
        self.VideoButton.grid(row=8, column=5, sticky=tk.E)


    def outlier_slider(self):
        self.low_threshold = tk.DoubleVar()
        self.high_threshold = tk.DoubleVar()
        self.outlierSlider =RangeSliderH(root, [self.low_threshold,self.high_threshold],Width = 400, Height = 65, min_val = 0, max_val = 1, show_value=True,padX=17
                                         , line_s_color="#7eb1c2",digit_precision='.2f')

        self.outlierSlider.grid(padx = 10, pady = 10, row=28, column=2,columnspan=1, sticky=tk.E)
        #self.show_threshold()
        self.low_rate_text_box()
        self.high_rate_text_box()

    def low_rate_text_box(self):
        self.low_rate = tk.StringVar()
        self.low_rate.set("0.2")
        tk.Label(root, text="below low threshold").grid(row=29, column=2)
        self.e1 = tk.Entry(root, textvariable=self.low_rate)
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
        self.ClearInterestedAreasButton = tk.Button(root, text='Clear_Rectangles', fg='#ffffff', bg='#999999',
                                               activebackground='#454545',
                                               relief=tk.RAISED, width=16,padx=10, pady=5,
                                               font=(self.widgetFont, self.widgetFontSize), command=self.clear_rects,
                                               )
        self.ClearInterestedAreasButton.grid(row=9, column=5,
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
        self.selAutoExposureLabel = tk.Label(root, text='Select AutoExposure:', font=(self.widgetFont, self.widgetFontSize))
        self.selAutoExposureLabel.grid(row=0, column=4, sticky=tk.W)
        self.AutoExposureList = tk.OptionMenu(root, self.defAutoExposure, *self.auto_exposures, command=self.setAutoExposure)
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
        self.c1 = tk.Checkbutton(root, text='Consider outliers at local selection', variable = self.local_consider_outliers_check_, offvalue=0, onvalue=1, command= self.switch_outlier)
        self.c1.grid(row = 23, column = 5)

    def high_res_checkbox(self):

        self.high_res_check = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text='High Resolution', variable = self.high_res_check, offvalue=0, onvalue=1, command= self.switch_res)
        self.c1.grid(row = 28, column = 1)

    def switch_outlier(self):
        self.local_consider_outliers_check = self.local_consider_outliers_check_.get()
        print("local_consider_outliers_check is ", self.local_consider_outliers_check)

    def switch_res(self):

        self.res_check = self.high_res_check.get()
        print("self.res_check is ", self.res_check)

    def switch_raw(self):

        self.useRawIms = self.useRawIms_.get()
        print("self.useRawIms is ", self.useRawIms)

    def mertens_checkbox(self):

        self.mertens_check = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text=' Mertens Export', variable= self.mertens_check, offvalue= 0, onvalue= 1, command= self.switch_mertens)
        self.c1.grid(row = 29, column = 1)

    def switch_mertens(self):

        self.hdr_mode_check = self.mertens_check.get()
        print("self.hdr_mode_check is ", self.hdr_mode_check)

    def horizontal_slider(self):
        # Horizantal Slider
        self.horSlider = tk.Scale(root, activebackground='black', cursor='sb_h_double_arrow', from_=0, to=self.frame_num[0] - 1,
                             label='Frame Number', font=(self.widgetFont, self.widgetFontSize), orient=tk.HORIZONTAL,
                             length=self.widthToScale, command=self.updateSlider)
        self.horSlider.grid(row=27, column=1, columnspan=2,  sticky=tk.SW)

    def vertical_slider(self):
        # Vertical Slider

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


        self.verSliderLabel = tk.Label(root, text='Exposure Time', font=(self.widgetFont, self.widgetFontSize))
        self.verSliderLabel.grid(row=0, column=0)

        # self.verSlider = tk.Scale(root, activebackground='black', cursor='sb_v_double_arrow', from_=0,
        #                      to=self.stack_size[self.scene_index] - 1, font=(self.widgetFont, self.widgetFontSize), length=self.heightToScale,
        #                      command=self.updateSlider)

        self.verSlider = tk.Scale(root, activebackground='black', cursor='sb_v_double_arrow', from_=min(self.SCALE_LABELS), to=max(self.SCALE_LABELS), font=(self.widgetFont, self.widgetFontSize),
                                  length=self.heightToScale,
                                  command= self.scale_labels)

        print(self.verSlider.configure().keys())

        self.verSlider.grid(row=1, column=0, rowspan=25)

    def scale_labels(self, value):

        # self.verSlider.config(label=self.SCALE_LABELS[int(value)])
        tk.Label(root, text=self.SCALE_LABELS[int(value)], font=("Times New Roman", 15)).grid(row=27, column=0, )

        # self.verSlider.place(x=50, y=300, anchor="center")
        self.useMertens = False


        if(self.current_auto_exposure == "None"):
            self.updateSlider(value)




        # scale = tk.Scale(root, from_=min(SCALE_LABELS), to=max(SCALE_LABELS),
        #                  orient=tk.HORIZONTAL, showvalue=False, command=scale_labels)



    def image_mean_plot(self,stack_size=15,curr_frame_mean_list=np.zeros(15),ind=0,val=0):
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
        plt.text(ind, val,'(' + str(ind) + ', ' + str("%.2f" % val) + ')', color='red',
                 fontsize=13, position=(0 - 0.2, val + 0.04))
        plt.title('Exposure stack mean')
        plt.xlabel('Image index')
        plt.ylabel('Mean value')
        plt.xlim(-0.2, stack_size - 0.8)
        if stack_size < 20:
            plt.xticks(np.arange(0, stack_size, 1))
        elif stack_size >= 15 and stack_size < 30:
            plt.xticks(np.arange(0, stack_size, 2))
        else:
            plt.xticks(np.arange(0, stack_size, 3))

        plt.ylim(-0.02, 0.85)
        plt.yticks(np.arange(0, 0.85, 0.1))
        self.fig.canvas.draw()

        self.tempImg_2 = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        self.photo_2 = ImageTk.PhotoImage(self.tempImg_2)
        self.imagePrevlabel_2 = tk.Label(root, image=self.photo_2)
        self.imagePrevlabel_2.grid(row=2, column=3, columnspan=2, rowspan=15, sticky=tk.NE)

    def hist_plot(self,count1=np.zeros(100),count2=np.zeros(100)):
        font = {'family': 'monospace',
                'weight': 'bold',
                'size': 10}
        bins = np.arange(1,self.num_bins+1)
        #self.fig = plt.figure(figsize=(4, 4))  # 4.6, 3.6
        if self.fig_2:
            plt.close(self.fig_2)
            self.fig_2.clear()
        self.fig_2, axes = plt.subplots(2, sharex=True, sharey=True,figsize=(4, 6))


        axes[1].bar(bins, count2, align='center')
        axes[0].bar(bins, count1, align='center')
        axes[1].set_title('histogram with outlier',**font)
        axes[0].set_title('histogram without outlier',**font)

        self.fig_2.canvas.draw()

        self.tempImg_3 = Image.frombytes('RGB', self.fig_2.canvas.get_width_height(), self.fig_2.canvas.tostring_rgb())
        self.photo_3 = ImageTk.PhotoImage(self.tempImg_3)
        self.imagePrevlabel_3 = tk.Label(root, image=self.photo_3)
        self.imagePrevlabel_3.grid(row=17, column=3, columnspan=2, rowspan=20, sticky=tk.NE)

    def hist_plot_three(self, count1=np.zeros(100), count2=np.zeros(100),stack_size=15,curr_frame_mean_list=np.zeros(15),ind=0,val=0):
        font = {'family': 'monospace',
                'weight': 'bold',
                'size': 10}
        bins = np.arange(1, self.num_bins + 1)
        # self.fig = plt.figure(figsize=(4, 4))  # 4.6, 3.6
        if self.fig_2:
            plt.close(self.fig_2)
            self.fig_2.clear()
        self.fig_2, axes = plt.subplots(3, figsize=(4, 9))
        self.fig_2.tight_layout()

        axes[1].bar(bins, count2, align='center')
        axes[0].bar(bins, count1, align='center')
        axes[1].set_title('histogram with outlier', **font)
        axes[0].set_title('histogram without outlier', **font)
        axes[1].sharex(axes[0])
        axes[1].sharey(axes[0])
        axes[2].plot(np.arange(stack_size), curr_frame_mean_list, color='green',
                     linewidth=2)  # ,label='Exposure stack mean')
        axes[2].plot(ind, val, color='red', marker='o', markersize=12)
        axes[2].text(ind, val, '(' + str(ind) + ', ' + str("%.2f" % val) + ')', color='red',
                 fontsize=13, position=(0 - 0.2, val + 0.01))
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

        #tempImg = Image.fromarray(temp_img)
        #self.photo = ImageTk.PhotoImage(tempImg)
        #self.imagePrevlabel.configure(image=self.photo)

    def HdrMedian(self):

        temp_img_ind = int(self.horSlider.get() * self.stack_size[self.scene_index])
        temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])

        temp_img = (np.median(temp_stack, axis=0)).astype(np.uint8)
        cv2.putText(temp_img, 'HDR-Median', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #tempImg = Image.fromarray(temp_img)
        #self.photo = ImageTk.PhotoImage(tempImg)
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
        cv2.putText(mean_min_dis_med, 'HDR-Abdullah', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        #tempImg = Image.fromarray(mean_min_dis_med)
        #self.photo = ImageTk.PhotoImage(tempImg)
        #self.imagePrevlabel.configure(image=self.photo)

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

            m1 = Image.fromarray(reg_vid[0])
            m2 = reg_vid_plot[0]
            sv = self.get_concat_h_blank(m1, m2)

            self.check_fps()

            fold_name = self.scene[self.scene_index] + "_0.12_Ex_" + list[int(self.verSlider.get())] + "_FPS_" + str(self.video_fps)
            folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
            os.makedirs(folderStore, exist_ok=True)
            connected_image = folderStore + self.joinPathChar + fold_name + ".avi"

            # capture the image and save it on the save path
            os.makedirs(folderStore, exist_ok=True)


            print(self.video_fps)
            video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.video_fps,
                                    (sv.width, sv.height))

            for i in range(len(reg_vid)):

                tempImg = Image.fromarray(reg_vid[i])
                temp_img_plot = reg_vid_plot[i]

                array = np.array(self.get_concat_h_blank(tempImg, temp_img_plot))
                video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

            video.release()

        elif self.res_check == 0 and self.current_auto_exposure == "Global" or self.current_auto_exposure == 'Local':
            if not len(self.eV) == 100:
                return

            for i in range(100):
                self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + self.eV[i]
                self.check = False
                self.updatePlot()
                reg_vid_plot.append(self.tempImg_2)

                img = deepcopy(self.img_all[self.temp_img_ind])
                reg_vid.append(img)

            m1 = Image.fromarray(reg_vid[0])
            m2 = reg_vid_plot[0]
            sv = self.get_concat_h_blank(m1, m2)

            self.check_fps()

            fold_name = self.scene[self.scene_index] + "_0.12_Ex_" + self.current_auto_exposure + "_FPS_" + str(
                self.video_fps)
            folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
            os.makedirs(folderStore, exist_ok=True)
            connected_image = folderStore + self.joinPathChar + fold_name + ".avi"

            # capture the image and save it on the save path
            os.makedirs(folderStore, exist_ok=True)

            #print(self.eV)
            video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.video_fps,
                                    (sv.width, sv.height))

            for i in range(len(reg_vid)):
                tempImg = Image.fromarray(reg_vid[i])
                temp_img_plot = reg_vid_plot[i]

                array = np.array(self.get_concat_h_blank(tempImg, temp_img_plot))
                video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

            video.release()

            self.check_fps()

        elif self.res_check == 1 and self.current_auto_exposure == "None":

            self.check_fps()

            regular.main(self.scene[self.scene_index], self.video_fps, self.verSlider.get(), list, self.folders)

        elif self.res_check == 1 and self.current_auto_exposure == "Global" or self.current_auto_exposure == 'Local':

            self.check_fps()

            high_res_auto_ex_video.main(self.scene[self.scene_index], self.video_fps,self.eV,self.current_auto_exposure, self.folders)

    def get_concat_h_blank(self, im1, im2, color=(0, 0, 0)):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def check_fps(self):

        print("text is ", self.save_video_fps.get())

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

        #print("row num girds is ", self.row_num_grids_.get())

        if self.validate_num_grids(self.row_num_grids_.get()) is True:

            try:
                self.row_num_grids = int(self.row_num_grids_.get())
                # print(set_speed)
            except ValueError:
                self.row_num_grids = 8  # set as default speed
        else:
            self.row_num_grids = 8

    def check_col_num_grids(self):

        #print("col num girds is ", self.col_num_grids_.get())

        if self.validate_num_grids(self.col_num_grids_.get()) is True:

            try:
                self.col_num_grids = int(self.col_num_grids_.get())
                # print(set_speed)
            except ValueError:
                self.col_num_grids = 8  # set as default speed
        else:
            self.col_num_grids = 8

    def pauseRun(self):
        print("in pause")
        self.play = False

    def runVideo(self):
        self.play = True
        self.playVideo()

    def setValues(self, dummy=False):

        # self.play = True
        # self.playVideo()
        # time.sleep(1)
        # print(scene_name)
        self.check_num_grids()
        if self.scene[self.scene_index] != self.defScene.get():
            self.clear_rects()
            self.scene_index = self.scene.index(self.defScene.get())
            #input_ims = 'Image_Arrays_exposure/Scene' + str(self.scene_index + 1) + '_ds_raw_imgs.npy'
            self.setAutoExposure()

            # exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"], r_percent=self.exposureParams['r_percent'], g_percent=self.exposureParams['g_percent'],
            #                                     col_num_grids=self.exposureParams['col_num_grids'], row_num_grids=self.exposureParams['row_num_grids'], low_threshold=self.exposureParams['low_threshold'], low_rate=self.exposureParams['low_rate'],
            #                                     high_threshold=self.exposureParams['high_threshold'], high_rate=self.exposureParams['high_rate'])
            # self.eV,weighted_means,hists,hists_before_ds_outlier = exposures.pipeline()

            self.img_all = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.defScene.get() + '_imgs_' + str(self.downscale_ratio) + '.npy')
            self.img_mean_list = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.defScene.get() + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2 ** self.bit_depth - 1)

            self.img_mertens = np.load(os.path.join(os.path.dirname(__file__), 'Image_Arrays') + self.joinPathChar + self.scene[self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')

            self.img_raw = np.load(
                os.path.join(os.path.dirname(__file__), 'Image_Arrays_from_dng') + self.joinPathChar + self.scene[
                    self.scene_index] + '_show_dng_imgs' + '.npy')

            self.resetValues()

    def setAutoExposure(self, dummy=False):
        self.current_auto_exposure = self.defAutoExposure.get()
        self.scene_index = self.scene.index(self.defScene.get())

        input_ims = 'Image_Arrays_exposure/Scene' + str(self.scene_index+1) + '_ds_raw_imgs.npy'
        self.check_num_grids()
        self.exposureParams = {"downsample_rate":1/25,'r_percent':0,'g_percent':1,
                                                'col_num_grids':self.col_num_grids, 'row_num_grids':self.row_num_grids, 'low_threshold':self.low_threshold.get(), 'low_rate':float(self.low_rate.get()),
                                                'high_threshold':self.high_threshold.get(), 'high_rate':float(self.high_rate.get())}
        if(self.current_auto_exposure == "Global"):
            self.clear_rects()
            exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"], r_percent=self.exposureParams['r_percent'], g_percent=self.exposureParams['g_percent'],
                                                col_num_grids=self.exposureParams['col_num_grids'], row_num_grids=self.exposureParams['row_num_grids'], low_threshold=self.exposureParams['low_threshold'], low_rate=self.exposureParams['low_rate'],
                                                high_threshold=self.exposureParams['high_threshold'], high_rate=self.exposureParams['high_rate'])
            #exposures = exposure_class.Exposure(params = self.exposureParams)
            self.eV,self.weighted_means,self.hists,self.hists_before_ds_outlier = exposures.pipeline()

        elif(self.current_auto_exposure == "Local"):
            self.clear_rects_local_wo_grids()
            consider_outliers = bool(self.local_consider_outliers_check)

            list_local = local_interested_grids_generater(self.row_num_grids, self.col_num_grids, self.rectangles)

            exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"], r_percent=self.exposureParams['r_percent'], g_percent=self.exposureParams['g_percent'],
                                                col_num_grids=self.exposureParams['col_num_grids'], row_num_grids=self.exposureParams['row_num_grids'], low_threshold=self.exposureParams['low_threshold'], low_rate=self.exposureParams['low_rate'],
                                                high_threshold=self.exposureParams['high_threshold'], high_rate=self.exposureParams['high_rate'],local_indices=list_local)
            self.eV,self.weighted_means,self.hists,self.hists_before_ds_outlier = exposures.pipeline()
        elif(self.current_auto_exposure == "Local without grids"):
            self.clear_rects_local()
            list_local = self.list_local_without_grids()

            exposures = exposure_class.Exposure(input_ims, downsample_rate=self.exposureParams["downsample_rate"],
                                                r_percent=self.exposureParams['r_percent'],
                                                g_percent=self.exposureParams['g_percent'],
                                                col_num_grids=self.exposureParams['col_num_grids'],
                                                row_num_grids=self.exposureParams['row_num_grids'],
                                                low_threshold=self.exposureParams['low_threshold'],
                                                low_rate=self.exposureParams['low_rate'],
                                                high_threshold=self.exposureParams['high_threshold'],
                                                high_rate=self.exposureParams['high_rate'], local_indices=list_local)
            self.eV,self.weighted_means,self.hists,self.hists_before_ds_outlier = exposures.pipeline_local_without_grids()

        print("CURRENT AUTO EXPOSURE", self.current_auto_exposure)

    def list_local_without_grids(self):
        list_ = []
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for (y_start,x_strat,y_end,x_end) in self.rects_without_grids:
            list_.append([y_start/h,x_strat/w,y_end/h,x_end/w])
        return list_


    def playVideo(self):
        # global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play, video_speed, useMertens

        #

        if self.validate_video_speed(self.video_speed.get()) is True:

            try:
                set_speed = int(self.video_speed.get())
                # print(set_speed)
            except ValueError:
                set_speed = 300  # set as default speed

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
        #if self.current_auto_exposure == "Local" or self.current_auto_exposure == "Local without grids":
        self.setAutoExposure()
        # self.useMertens = False
        # print("Reset")
        self.play = False
        # verSlider.config(to=stack_size[scene_index]-1)
        self.horSlider.config(to=self.frame_num[self.scene_index] - 1)
        # verSlider.set(0),
        self.horSlider.set(0)

       # self.imagePrevlabel.configure(image=photo)
        self.updatePlot()

    def updatePlot(self):
        # global verSlider, horSlider, photo, photo_2, stack_size, img_all, img, img_mean_list, scene_index, fig

        if self.check == True:
            self.temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + int(self.verSlider.get())
        else:
            pass

        self.check == True
        # Image mean plot
        if len(self.hists) != 0:
            first_ind = self.temp_img_ind // self.stack_size[self.scene_index]
            send_ind = self.temp_img_ind % self.stack_size[self.scene_index]
            count1 = self.hists[first_ind][send_ind]
            count2 = self.hists_before_ds_outlier[first_ind][send_ind]
            stack_size = self.stack_size[self.scene_index]
            curr_frame_mean_list = self.weighted_means[first_ind]
            ind = send_ind
            val = curr_frame_mean_list[send_ind]
        else:
            count1 = np.zeros(self.num_bins)
            count2 = np.zeros(self.num_bins)
            stack_size = 15
            curr_frame_mean_list = np.zeros(15)
            ind = 0
            val = 0
       # self.hist_plot_unvisible()
        #self.image_mean_plot(stack_size=stack_size,curr_frame_mean_list=curr_frame_mean_list,ind=ind,val=val)
        #self.hist_plot(count1=count1, count2=count2)
        self.hist_plot_three(count1=count1, count2=count2,stack_size=stack_size,curr_frame_mean_list=curr_frame_mean_list,ind=ind,val=val)

    def clear_rects(self):
        self.clear_rects_local()
        self.clear_rects_local_wo_grids()

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


    def mouse_events(self,event):
        if self.current_auto_exposure == "Local":
            self.canvas_click(event)
        if self.current_auto_exposure == "Local without grids":
            self.local_wo_grids(event)


    def canvas_click(self, event):
        col, row = event.x, event.y

        #self.clear_rects()

        if self.current_auto_exposure  == "Local":
            self.check_num_grids()
            self.colGridSelect = int(col * self.col_num_grids / self.photo.width())
            self.rowGridSelect = int(row * self.row_num_grids / self.photo.height())
            rect = [self.rowGridSelect, self.colGridSelect]
            self.rectangles.append(rect) #making this array to allow us to be flexible in the future
            self.current_rects.append(self.draw_rectangle(rect[0], rect[1], "green"))
            self.setAutoExposure()

    def draw_rectangle(self, row, col, color):
        ww = self.photo.width()
        hh= self.photo.height()
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
        #print(topx, topy, botx, boty)
        rect = self.canvas.create_rectangle(topx, topy, botx, boty, fill='', outline=color)
        return rect

    def local_wo_grids(self,event):
        self.on_button_press(event)
        self.on_move_press(event)
        self.on_button_release(event)

    def on_button_press(self, event):
        print("herehere")
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.current_auto_exposure == "Local without grids":
            # save mouse drag start position
            # create rectangle if not yet exist
            self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='red')
            self.current_rects_wo_grids.append(self.rect)

    def on_move_press(self, event):
        if self.current_auto_exposure == "Local without grids":
            curX = self.canvas.canvasx(event.x)
            curY = self.canvas.canvasy(event.y)

            w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
            print('w: '+str(w))
            print('h: '+str(h))
            print(event.x)
            print(event.y)

            # expand rectangle as you drag the mouse

            self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
            self.curX = curX
            self.curY = curY
            print("curx: "+ str(curX))
            print("cury: "+ str(curY))

    def on_button_release(self, event):
        print("rect: "+str(self.rect))
        print("start_x: "+str(self.start_x))
        print("start_y: "+str(self.start_y))
        print("cur_x: "+str(self.curX))
        print("cur_y: "+str(self.curY))
        if self.current_auto_exposure == "Local without grids":
            self.rects_without_grids.append([self.start_y, self.start_x, self.curY, self.curX])
            print(self.rects_without_grids)
        if self.current_auto_exposure  == "Local":
            self.check_num_grids()
            self.colGridSelect = int(self.start_x * self.col_num_grids / self.photo.width())
            self.rowGridSelect = int(self.start_y * self.row_num_grids / self.photo.height())
            rect = [self.rowGridSelect, self.colGridSelect]
            self.rectangles.append(rect) #making this array to allow us to be flexible in the future
            self.current_rects.append(self.draw_rectangle(rect[0], rect[1], "green"))
            self.setAutoExposure()

    def updateSlider(self, scale_value):
        if(self.current_auto_exposure != "None"):
            self.verSlider.set(self.eV[self.horSlider.get()])
            temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + self.eV[self.horSlider.get()]
        else:
            temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + int(self.verSlider.get())

        autoExposureMode = True

        # global verSlider, horSlider, photo, photo_2, imagePrevlabel, imagePrevlabel_2, img_all, img, img_mean_list, scene_index, fig, useMertens, mertensVideo
        #
        print(self.useRawIms)
        if self.useMertens:
            # img = self.mertensVideo[self.horSlider.get()]
            img = self.img_mertens[self.horSlider.get()]
        elif self.useRawIms:
            #print(self.verSlider.get())
            img = self.img_raw[self.horSlider.get()][self.verSlider.get()]
        else:
            img = deepcopy(self.img_all[temp_img_ind])

        # self.photo = ImageTk.PhotoImage(Image.fromarray(self.img))
        # self.imagePrevlabel = tk.Label(root, image=self.photo)
        # self.imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)

        tempImg = Image.fromarray(img).resize((self.canvas.winfo_width(),self.canvas.winfo_height()))



        self.photo = ImageTk.PhotoImage(tempImg,width=self.canvas.winfo_width(),height=self.canvas.winfo_height())
         #= self.photo  # Keep reference in case this code is put into a function.

        self.canvas.itemconfig(self.canvas_img, image=self.photo)

        self.canvas.tag_lower(self.canvas_img)

        # Keep reference in case this code is put into a function.
        self.updatePlot()



b = Browser(root)

root.mainloop()