from RangeSlider import RangeSliderH
from browser_builder import Browser

import tkinter as tk
import numpy as np
import update_visulization


class BrowserWithInputs(Browser):
    def __init__(self, root):
        super().__init__(root)
        self.smoothness_number = None
        self.start_index = None
        self.outlierSlider = None
        self.high_threshold = None
        self.low_threshold = None
        self.e1 = None
        self.target_intensity = None

    def init_functions(self):
        super().init_functions()
        self.target_intensity_text_box()
        self.smoothness_number_text_box()
        self.outlier_slider()
        update_visulization.hist_plot_three(self, stack_size=self.stack_size[0],
                                            curr_frame_mean_list=np.zeros(self.stack_size[0]))

    def smoothness_number_text_box(self):
        self.smoothness_number = tk.IntVar()
        self.smoothness_number.set(3)
        tk.Label(self.root, text="Smooth the results with n frames\nEnter an integer between 1 and 100").grid(row=29, column=2, sticky=tk.NSEW)
        self.e1 = tk.Entry(self.root, textvariable=self.smoothness_number)
        self.e1.grid(row=30, column=2)

    def target_intensity_text_box(self):
        self.target_intensity = tk.DoubleVar()
        self.target_intensity.set(0.13)
        tk.Label(self.root, text="target intensity").grid(row=28, column=1, sticky=tk.NSEW)
        self.e1 = tk.Entry(self.root, textvariable=self.target_intensity)
        self.e1.grid(row=29, column=1)

    def outlier_slider(self):
        self.low_threshold = tk.DoubleVar()
        self.high_threshold = tk.DoubleVar()
        self.outlierSlider = RangeSliderH(self.root, [self.low_threshold, self.high_threshold], Width=400, Height=65,
                                          min_val=0, max_val=1, show_value=True, padX=25
                                          , line_s_color="#7eb1c2", digit_precision='.2f')
        self.outlierSlider.grid(padx=8, pady=5, row=32, column=1, columnspan=2, sticky=tk.E)
        self.start_index_text_box()

    def start_index_text_box(self):
        self.start_index = tk.StringVar()
        self.start_index.set("15")
        tk.Label(self.root, text="start index").grid(row=30, column=1)
        self.e1 = tk.Entry(self.root, textvariable=self.start_index)
        self.e1.grid(row=31, column=1)
