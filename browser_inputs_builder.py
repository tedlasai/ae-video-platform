from RangeSlider import RangeSliderH

from browser_builder import Browser
import tkinter as tk

class Broswer_with_inputs(Browser):
    def __init__(self, root):
        super().__init__(root)

    def init_functions(self):
        super().init_functions()
        self.playback_text_box()
        self.video_fps_text()
        self.target_intensity_text_box()
        self.outlier_slider()
        self.local_consider_outliers_checkbox()
        self.number_of_previous_frames_text_box()
        self.stepsize_limit_text_box()
        self.local_interested_name_text_box()
        self.local_interested_global_area_percentage_box()
        self.show_SRGB_hist_check_box()

    def local_interested_name_text_box(self):
        self.local_interested_name = tk.StringVar()
        self.local_interested_name.set("")
        tk.Label(self.root, text="Name of Interested").grid(row=10, column=5)
        self.e1 = tk.Entry(self.root, textvariable=self.local_interested_name)
        self.e1.grid(row=11, column=5, sticky=tk.E)

    def target_intensity_text_box(self):
        self.target_intensity = tk.DoubleVar()
        self.target_intensity.set(0.13)
        tk.Label(self.root, text="target intensity").grid(row=29, column=5)
        self.e1 = tk.Entry(self.root, textvariable=self.target_intensity)
        self.e1.grid(row=30, column=5, sticky=tk.E)


    def local_interested_global_area_percentage_box(self):
        self.local_interested_global_area_percentage = tk.DoubleVar()
        self.local_interested_global_area_percentage.set(0.0)
        tk.Label(self.root, text="Global percentage on local selection").grid(row=25, column=5)
        self.e1 = tk.Entry(self.root, textvariable=self.local_interested_global_area_percentage)
        self.e1.grid(row=26, column=5, sticky=tk.E)

    def show_SRGB_hist_check_box(self):
        self.show_srgb_hist_check_ = tk.IntVar()
        self.c1 = tk.Checkbutton(self.root, text='Show SRGB Hist', variable=self.show_srgb_hist_check_, offvalue=0, onvalue=1,
                                 command=self.switch_SRGB_Hist)
        self.c1.grid(row=27, column=5)

    def number_of_previous_frames_text_box(self):
        self.number_of_previous_frames = tk.IntVar()
        self.number_of_previous_frames.set(3)
        tk.Label(self.root, text="# of previous frames").grid(row=31, column=5)
        self.e1 = tk.Entry(self.root, textvariable=self.number_of_previous_frames)
        self.e1.grid(row=32, column=5, sticky=tk.E)

    def stepsize_limit_text_box(self):
        self.stepsize_limit = tk.IntVar()
        self.stepsize_limit.set(5)
        tk.Label(self.root, text="step size limitation").grid(row=33, column=5)
        self.e1 = tk.Entry(self.root, textvariable=self.stepsize_limit)
        self.e1.grid(row=34, column=5, sticky=tk.E)


    def outlier_slider(self):
        self.low_threshold = tk.DoubleVar()
        self.high_threshold = tk.DoubleVar()
        #print("BUILDING OUTLIER SLIDER", self.high_threshold)
        self.outlierSlider = RangeSliderH(self.root, [self.low_threshold, self.high_threshold], Width=400, Height=65,
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
        tk.Label(self.root, text="start index").grid(row=29, column=2)
        self.e1 = tk.Entry(self.root, textvariable=self.start_index)
        self.e1.grid(row=30, column=2)

    def high_rate_text_box(self):
        self.high_rate = tk.StringVar()
        self.high_rate.set("0.2")
        tk.Label(self.root, text="above high threshold").grid(row=31, column=2)
        self.e1 = tk.Entry(self.root, textvariable=self.high_rate)
        self.e1.grid(row=32, column=2)


    def playback_text_box(self):
        # TextBox
        self.video_speed = tk.StringVar()
        # video_speed = 1
        tk.Label(self.root, text="Browser Playback Speed (ms delay)").grid(row=30, column=1)
        self.e1 = tk.Entry(self.root, textvariable=self.video_speed)
        self.e1.grid(row=31, column=1)

    def video_fps_text(self):
        # TextBox
        self.save_video_fps = tk.StringVar()
        # video_speed = 1
        tk.Label(self.root, text="Video FPS").grid(row=32, column=1)
        self.e1 = tk.Entry(self.root, textvariable=self.save_video_fps)
        self.e1.grid(row=33, column=1)

    def local_consider_outliers_checkbox(self):
        self.local_consider_outliers_check_ = tk.IntVar()
        self.c1 = tk.Checkbutton(self.root, text='Consider outliers at local selection',
                                 variable=self.local_consider_outliers_check_, offvalue=0, onvalue=1,
                                 command=self.switch_outlier)
        self.c1.grid(row=23, column=5)

    def switch_outlier(self):
        self.local_consider_outliers_check = self.local_consider_outliers_check_.get()
        print("local_consider_outliers_check is ", self.local_consider_outliers_check)


    def switch_SRGB_Hist(self):
        pass
        # self.show_srgb_hist_check = self.show_srgb_hist_check_.get()
        # print("self.show_srgb_hist_chec is ", self.show_srgb_hist_check)
        # self.updateSlider(0)







