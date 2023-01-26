from browser_builder import Browser

class Broswer_with_inputs(Browser):
    def __init__(self, root):
        super().__init__(root)

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





