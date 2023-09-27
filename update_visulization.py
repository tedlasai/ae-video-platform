from copy import deepcopy
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import button_functions
import matplotlib.pyplot as plt


def current_frame(b):
    result = b.img_all[b.horSlider.get()]
    return result


def scale_labels(b, _):
    text_ = b.SCALE_LABELS_NEW[0]
    tk.Label(b.root, text=text_, font=("Times New Roman", 15)).grid(row=27, column=0, )
    # temp_img_ind = int(b.horSlider.get()) * b.stack_size[0] + int(b.verSlider.get())
    updateHorSlider(b, _)


def updateSlider(b, _):
    if ((b.current_auto_exposure != "None") and (len(b.eV) > 0)):
        print(len(b.eV))
        b.verSlider.set(b.eV[b.horSlider.get()])
    #     temp_img_ind = int(b.horSlider.get()) * b.stack_size[b.scene_index] + b.eV[b.horSlider.get()]
    # else:
    #     temp_img_ind = int(b.horSlider.get()) * b.stack_size[0] + int(b.verSlider.get())

    updateHorSlider(b, _)


def updateHorSlider(b, _):
    autoExposureMode = True
    # if b.useRawIms:
    # # print(self.verSlider.get())
    #     img = b.img_raw[b.horSlider.get()][b.verSlider.get()]
    #
    # else:
    img = deepcopy(b.img_all[b.horSlider.get()][b.verSlider.get()])
    tempImg = Image.fromarray(img).resize((b.canvas.winfo_width(), b.canvas.winfo_height()))

    b.photo = ImageTk.PhotoImage(tempImg, width=b.canvas.winfo_width(), height=b.canvas.winfo_height())
    b.canvas.itemconfig(b.canvas_img, image=b.photo)

    b.canvas.tag_lower(b.canvas_img)
    # image_mean_plot()


def updatePlot(self):
    # global verSlider, horSlider, photo, photo_2, stack_size, img_all, img, img_mean_list, scene_index, fig
    stack_size = 40
    if self.check == True:
        self.temp_img_ind = int(self.horSlider.get()) * stack_size + int(self.verSlider.get())
    else:
        pass

    self.check = True
    # Image mean plot
    if len(self.hists) != 0:

        first_ind = round(self.temp_img_ind // stack_size)
        send_ind = round(self.temp_img_ind % stack_size)
        print(f'first_ind:{first_ind}')
        print(f'second_ind:{send_ind}')
        count1 = self.hists[first_ind][send_ind]
        # print("current srgb hist check")
        # print(self.show_srgb_hist_check)
        if self.show_srgb_hist_check == 1:
            # print("here")
            count2, self.srgb_mean = show_srgb_hist(self)
            # print(count2)
        else:
            count2 = self.hists_before_ds_outlier[first_ind][send_ind]
        # print(f'first_ind:{first_ind}')
        curr_frame_mean_list = self.weighted_means[first_ind]
        ind = send_ind
        val = curr_frame_mean_list[send_ind]
        ind2 = int(self.eV[self.horSlider.get()])
        val2 = curr_frame_mean_list[ind2]
        count3 = self.hists[first_ind][ind2]
        print("---")
        print(f'first_ind:{first_ind}')
        print(f'ind2:{ind2}')

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
    hist_plot_three(self, count1=count1, count2=count2, count3=count3, stack_size=stack_size,
                    curr_frame_mean_list=curr_frame_mean_list, ind=ind, val=val, ind2=ind2, val2=val2)
    if self.current_auto_exposure == "Local on moving objects":
        draw_interested_moving_areas_per_frame(self)


def draw_interested_moving_areas_per_frame(self):
    w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
    curr_frame = self.horSlider.get()
    print("length of moving areas")
    print(len(self.the_moving_area_list))
    try:
        rect_coordis = self.the_moving_area_list[curr_frame]
        button_functions.clear_moving_rects(self)
        for coords in rect_coordis:
            a, b, c, d = coords[1] * w, coords[0] * h, coords[3] * w, coords[2] * h
            rect = self.canvas.create_rectangle(a, b, c, d, outline='violet')
            self.moving_rectids.append(rect)
    except:
        pass


def hist_laxis(data, n_bins,
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
    current_rgb_img = deepcopy(self.img_all[self.horSlider.get()][self.verSlider.get()])
    current_rgb_img = current_rgb_img / (2 ** 8 - 1)
    current_rgb_img[:, :, 0] = current_rgb_img[:, :, 0] * 0.2126
    current_rgb_img[:, :, 1] = current_rgb_img[:, :, 1] * 0.7152
    current_rgb_img[:, :, 2] = current_rgb_img[:, :, 2] * 0.0722
    current_rgb_img_ = np.sum(current_rgb_img, axis=2)

    if self.current_auto_exposure == "Local on moving objects" and len(self.the_moving_area_list) > 0:
        interested_boundaries = self.the_moving_area_list[self.horSlider.get()]
        temp_img = np.ones(current_rgb_img_.shape) * (-0.01)
        h, w = current_rgb_img_.shape
        # w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        for coord in interested_boundaries:
            w_start = int(coord[1] * w)
            h_start = int(coord[0] * h)
            w_end = min(int(coord[3] * w) + 1, w + 1)
            h_end = min(int(coord[2] * h) + 1, h + 1)
            temp_img[h_start:h_end, w_start:w_end] = current_rgb_img_[h_start:h_end, w_start:w_end]
        temp_img = temp_img.flatten()
        srgb_hist, dropped = self.get_hists(temp_img)
        mean = self.get_means(dropped, temp_img)
    else:
        mean = np.mean(current_rgb_img_)
        current_rgb_img_ = current_rgb_img_.flatten()
        srgb_hist = hist_laxis(current_rgb_img_, self.num_bins + 1,
                               (0, 1.01))
    return srgb_hist, mean


def get_hists(self, flatten_weighted_ims):
    scene_hists_include_drooped_counts = hist_laxis(flatten_weighted_ims, self.num_bins + 2, (
        -0.01, 1.01))  # one extra bin is used to count the number of -0.01
    num_dropped_pixels = scene_hists_include_drooped_counts[0]
    scene_hists = scene_hists_include_drooped_counts[1:]
    return scene_hists, num_dropped_pixels


def get_means(num_dropped_pixels, flatten_weighted_ims):
    weighted_all_means = np.mean(flatten_weighted_ims)
    if num_dropped_pixels == 0:
        return weighted_all_means
    c = len(flatten_weighted_ims)
    mean = (c * weighted_all_means + 0.01 * num_dropped_pixels) / (c - num_dropped_pixels)
    return mean


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

        axes[1].set_title('sRGB image histogram', **font)
        axes[1].text(70, 0.2, '( mean: ' + str("%.2f" % self.srgb_mean) + ')', color='blue',
                     fontsize=13, position=(70, 0.2))
        axes[0].set_title('RAW image histogram', **font)
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
    self.imagePrevlabel_3 = tk.Label(self.root, image=self.photo_3)
    self.imagePrevlabel_3.grid(row=2, column=4, columnspan=2, rowspan=34, padx=20, pady=10, sticky=tk.NE)
