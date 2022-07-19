import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mp
import os
import glob

mp.rcParams.update({'axes.titlesize': 14, 'font.size': 11, 'font.family': 'arial'})

#Tkinter Window
root=tk.Tk()
root.geometry('1600x900'), root.title('Data Browser') #1900x1000+5+5

class Browser:

    def __init__(self, root):
        super().__init__()
        # myB = Frame(master)
        # myB.pack()
        self.widgetFont = 'Arial'
        self.widgetFontSize = 12
        self.scene = ['Scene101', 'Scene102', 'Scene103', 'Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
                      'Scene7', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15', 'Scene16', 'Scene17', 'Scene18']
        self.frame_num = [90, 65, 15, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # number of frames per position
        self.stack_size = [12, 47, 28, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]  # number of shutter options per position

        self.scene = ['Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15',
                      'Scene16', 'Scene17', 'Scene18']
        self.frame_num = [100, 100, 100, 100, 100, 100,
                          100, 100]  # number of frames per position
        self.stack_size = [15, 15, 15, 15, 15, 15, 15,
                           15]  # number of shutter options per position

        self.scene_index = 0
        self.mertensVideo = []
        self.bit_depth = 8
        self.downscale_ratio = 0.12
        self.check = True
        self.temp_img_ind = 0

        self.imgSize = [int(4480 * self.downscale_ratio), int(6720 * self.downscale_ratio)]
        self.widthToScale = self.imgSize[1]
        self.widPercent = (self.widthToScale / float(self.imgSize[1]))
        self.heightToScale = int(float(self.imgSize[0]) * float(self.widPercent))

        self.img_all = np.load(self.scene[self.scene_index] + '_imgs_' + str(self.downscale_ratio) + '.npy')
        self.img_mean_list = np.load(self.scene[self.scene_index] + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2**self.bit_depth - 1)
        self.img_mertens = np.load(self.scene[self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')

        self.img = deepcopy(self.img_all[0])
        self.useMertens = False
        self.play = True
        self.video_speed = 50
        self.regular_video_fps = 30

        self.video_high_res_check = 0

        # Image Convas
        self.photo = ImageTk.PhotoImage(Image.fromarray(self.img))
        self.imagePrevlabel = tk.Label(root, image=self.photo)
        self.imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)

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
        self.video_fps()
        self.horizontal_slider()
        self.vertical_slider()
        self.image_mean_plot()
        self.regular_video_button()
        self.high_res_checkbox()
        self.mertens_checkbox()

    def hdr_mean_button(self):
        # HDR Button - Mean
        self.HdrMeanButton = tk.Button(root, text='HDR-Mean', fg='#ffffff', bg='#999999', activebackground='#454545',
                                  relief=tk.RAISED, width=16, font=(self.widgetFont, self.widgetFontSize), command=self.HdrMean)
        self.HdrMeanButton.grid(row=29, column=2, sticky=tk.E)  # initial row was 26, +1 increments for all other rows

    def hdr_median_button(self):
        # HDR Button - Median
        self.HdrMedianButton = tk.Button(root, text='HDR-Median', fg='#ffffff', bg='#999999', activebackground='#454545',
                                    relief=tk.RAISED, width=16, font=(self.widgetFont, self.widgetFontSize), command=self.HdrMedian)
        self.HdrMedianButton.grid(row=30, column=2, sticky=tk.E)

    def hdr_mertens_button(self):
        # HDR Button - Mertens
        self.HdrMertensButton = tk.Button(root, text='HDR-Mertens', fg='#ffffff', bg='#999999', activebackground='#454545',
                                     relief=tk.RAISED, width=16, font=(self.widgetFont, self.widgetFontSize), command=self.HdrMertens)
        self.HdrMertensButton.grid(row=31, column=2, sticky=tk.E)

    def hdr_abdullah_button(self):
        # HDR Button - Abdullah
        self.HdrAbdullahButton = tk.Button(root, text='HDR-Abdullah', fg='#ffffff', bg='#999999', activebackground='#454545',
                                      relief=tk.RAISED, width=16, font=(self.widgetFont, self.widgetFontSize),
                                      command=self.HdrAbdullah)
        self.HdrAbdullahButton.grid(row=32, column=2, sticky=tk.E)

    def hdr_run_button(self):
        # Run Button
        self.RunButton = tk.Button(root, text='Run', fg='#ffffff', bg='#999999', activebackground='#454545',
                              relief=tk.RAISED, width=16, font=(self.widgetFont, self.widgetFontSize), command=self.setValues)
        self.RunButton.grid(row=33, column=2, sticky=tk.E)

    def hdr_pause_button(self):
        self.PauseButton = tk.Button(root, text='Pause', fg='#ffffff', bg='#999999', activebackground='#454545',
                                relief=tk.RAISED,
                                width=16, font=(self.widgetFont, self.widgetFontSize), command=self.pauseRun)
        self.PauseButton.grid(row=34, column=2, sticky=tk.E)

    def hdr_reset_button(self):
        # Reset Button
        self.RestButton = tk.Button(root, text='Reset', fg='#ffffff', bg='#999999', activebackground='#454545',
                               relief=tk.RAISED, width=16, font=(self.widgetFont, self.widgetFontSize), command=self.resetValues)
        self.RestButton.grid(row=35, column=2, sticky=tk.E)

    def regular_video_button(self):

        self.VideoButton = tk.Button(root, text='Video', fg='#ffffff', bg='#999999', activebackground='#454545',
                                relief=tk.RAISED,
                                width=16, font=(self.widgetFont, self.widgetFontSize), command=self.regular_video)
        self.VideoButton.grid(row=37, column=2, sticky=tk.E)

    def scene_select(self):
        # Select Scene List
        self.defScene = tk.StringVar(root)
        self.defScene.set(self.scene[self.scene_index])  # default value
        self.selSceneLabel = tk.Label(root, text='Select Scene:', font=(self.widgetFont, self.widgetFontSize))
        self.selSceneLabel.grid(row=0, column=2, sticky=tk.W)
        self.sceneList = tk.OptionMenu(root, self.defScene, *self.scene, command=self.setValues)
        self.sceneList.config(font=(self.widgetFont, self.widgetFontSize - 2), width=15, anchor=tk.W)
        self.sceneList.grid(row=1, column=2, sticky=tk.NE)

    def playback_text_box(self):
        # TextBox
        self.video_speed = tk.StringVar()
        # video_speed = 1
        tk.Label(root, text="Browser Playback Speed (FPS)").grid(row=34, column=1)
        self.e1 = tk.Entry(root, textvariable=self.video_speed)
        self.e1.grid(row=35, column=1)

    def video_fps(self):
        # TextBox
        self.save_video_fps = tk.StringVar()
        # video_speed = 1
        tk.Label(root, text="Video FPS").grid(row=36, column=1)
        self.e1 = tk.Entry(root, textvariable=self.save_video_fps)
        self.e1.grid(row=37, column=1)

    def high_res_checkbox(self):

        self.c1 = tk.Checkbutton(root, text='High Resolution', onvalue=1, offvalue=0, command= self.switch_res)
        self.c1.grid(row = 33, column = 1)

    def mertens_checkbox(self):

        self.mertens_high_res_check = tk.IntVar()
        self.c1 = tk.Checkbutton(root, text=' Mertens Export', variable= self.mertens_high_res_check, offvalue= 0, onvalue= 1, command= self.switch_mertens_res)
        self.c1.grid(row = 32, column = 1)

    def switch_mertens_res(self):

        self.video_high_res_check = self.mertens_high_res_check.get()
        print(self.video_high_res_check)


    def horizontal_slider(self):
        # Horizantal Slider
        self.horSlider = tk.Scale(root, activebackground='black', cursor='sb_h_double_arrow', from_=0, to=self.frame_num[0] - 1,
                             label='Frame Number', font=(self.widgetFont, self.widgetFontSize), orient=tk.HORIZONTAL,
                             length=self.widthToScale, command=self.updateSlider)
        self.horSlider.grid(row=31, column=1, sticky=tk.SW)

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

        self.verSlider.grid(row=1, column=0, rowspan=30)

    def scale_labels(self, value):

        # self.verSlider.config(label=self.SCALE_LABELS[int(value)])
        tk.Label(root, text=self.SCALE_LABELS[int(value)], font=("Times New Roman", 15)).grid(row=31, column=0, )

        # self.verSlider.place(x=50, y=300, anchor="center")
        self.useMertens = False
        self.updateSlider(value)


        # scale = tk.Scale(root, from_=min(SCALE_LABELS), to=max(SCALE_LABELS),
        #                  orient=tk.HORIZONTAL, showvalue=False, command=scale_labels)

    def image_mean_plot(self):

        self.fig = plt.figure(figsize=(4, 4))  # 4.6, 3.6
        plt.plot(np.arange(self.stack_size[self.scene_index]), self.img_mean_list[0:self.stack_size[self.scene_index]], color='green',
                 linewidth=2)  # ,label='Exposure stack mean')
        plt.plot(0, self.img_mean_list[0], color='red', marker='o', markersize=12)
        plt.text(0, self.img_mean_list[0], '(' + str(0) + ', ' + str("%.2f" % self.img_mean_list[0]) + ')', color='red',
                 fontsize=13, position=(0 - 0.2, self.img_mean_list[0] + 0.04))
        plt.title('Exposure stack mean')
        plt.xlabel('Image index')
        plt.ylabel('Mean value')
        plt.xlim(-0.2, self.stack_size[self.scene_index] - 0.8)
        if self.stack_size[self.scene_index] < 20:
            plt.xticks(np.arange(0, self.stack_size[self.scene_index], 1))
        elif self.stack_size[self.scene_index] >= 15 and self.stack_size[self.scene_index] < 30:
            plt.xticks(np.arange(0, self.stack_size[self.scene_index], 2))
        else:
            plt.xticks(np.arange(0, self.stack_size[self.scene_index], 3))

        plt.ylim(-0.02, 0.85)
        plt.yticks(np.arange(0, 0.85, 0.1))
        self.fig.canvas.draw()

        self.tempImg_2 = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        self.photo_2 = ImageTk.PhotoImage(self.tempImg_2)
        self.imagePrevlabel_2 = tk.Label(root, image=self.photo_2)
        self.imagePrevlabel_2.grid(row=4, column=2, columnspan=2, rowspan=24, sticky=tk.NE)

    def HdrMean(self):

        temp_img_ind = int(self.horSlider.get() * self.stack_size[self.scene_index])
        temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])

        temp_img = (np.mean(temp_stack, axis=0)).astype(np.uint8)
        cv2.putText(temp_img, 'HDR-Mean', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        tempImg = Image.fromarray(temp_img)
        self.photo = ImageTk.PhotoImage(tempImg)
        self.imagePrevlabel.configure(image=self.photo)

    def HdrMedian(self):

        temp_img_ind = int(self.horSlider.get() * self.stack_size[self.scene_index])
        temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])

        temp_img = (np.median(temp_stack, axis=0)).astype(np.uint8)
        cv2.putText(temp_img, 'HDR-Median', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        tempImg = Image.fromarray(temp_img)
        self.photo = ImageTk.PhotoImage(tempImg)
        self.imagePrevlabel.configure(image=self.photo)

    def HdrMertens(self):

        self.mertensVideo = []
        self.useMertens = True
        self.mertens_pic = []

        self.updateSlider(0)

        # for i in range(self.frame_num[self.scene_index]):
        #     temp_img_ind = int(i * self.stack_size[self.scene_index])
        #     temp_stack = deepcopy(self.img_all[temp_img_ind:temp_img_ind + self.stack_size[self.scene_index]])
        #
        #     # Exposure fusion using Mertens
        #     merge_mertens = cv2.createMergeMertens()
        #     res_mertens = merge_mertens.process(temp_stack)
        #     print(type(res_mertens))
        #
        #     # print(type(res_mertens))
        #     # Convert datatype to 8-bit and save
        #
        #     res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
        #     cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #
        #     img = Image.fromarray(res_mertens_8bit)
        #     # im.save("your_file_" + str(i) + ".jpeg")
        #
        #     # print(type(res_mertens_8bit))
        #
        #     print(i)
        #     self.mertensVideo.append(res_mertens_8bit)
        #     height, width, layers = 600, 800, 3 #img.shape
        #     size = (width, height)
        #     self.mertens_pic.append(img)
        #
        # self.check_fps()
        #
        # vid_name = self.scene[self.scene_index] + "_0.12_" + "Mertens" + "_FPS_" + str(self.regular_video_fps) + ".avi"
        # folderStore = os.path.join(os.path.dirname(__file__), 'HDR_Mertens_Video')
        # os.makedirs(folderStore, exist_ok=True)
        # save_vid = folderStore + '\\' + vid_name
        #
        # video = cv2.VideoWriter(save_vid, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.regular_video_fps,
        #                         (806, 538))  # fourcc,
        #
        #
        # print(folderStore)
        # # capture the image and save it on the save path
        # # os.makedirs(folderStore, exist_ok=True)
        #
        # for i in range(len(self.mertensVideo)):
        #     # tempImg = Image.fromarray(self.mertensVideo[i])
        #     # print(type(mertensVideo[i]))
        #     # save_image = folderStore + '\\' + fold_name + "_" + str(i) + ".jpeg"
        #
        #     # tempImg.save(save_image)
        #     video.write(cv2.cvtColor(self.mertensVideo[i], cv2.COLOR_RGB2BGR))
        #     print(type(self.mertensVideo[i]))
        #
        # video.release()
        #
        # print("mertens finished")



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

        tempImg = Image.fromarray(mean_min_dis_med)
        self.photo = ImageTk.PhotoImage(tempImg)
        self.imagePrevlabel.configure(image=self.photo)

    def regular_video(self):

        reg_vid = []
        reg_vid_plot = []
        for i in range(100):
            self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + int(self.verSlider.get())

            self.check = False
            self.updatePlot()
            reg_vid_plot.append(self.tempImg_2)

            img = deepcopy(self.img_all[self.temp_img_ind])
            reg_vid.append(img)


        list = ['15', '8', '6', '4', '2', '1', '05', '1-4', '1-8', '1-15', '1-30', '1-60', '1-125', '1-250', '1-500']

        m1 = Image.fromarray(reg_vid[0])
        m2 = reg_vid_plot[0]
        sv = self.get_concat_h_blank(m1, m2)

        self.check_fps()

        fold_name = self.scene[self.scene_index] + "_" + list[int(self.verSlider.get())] + "_FPS_" + str(self.regular_video_fps)
        folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
        connected_image = folderStore + '\\' + fold_name + ".avi"

        # capture the image and save it on the save path
        os.makedirs(folderStore, exist_ok=True)


        print(self.regular_video_fps)
        video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.regular_video_fps,
                                (sv.width, sv.height))

        for i in range(len(reg_vid)):
            tempImg = Image.fromarray(reg_vid[i])

            temp_img_plot = reg_vid_plot[i]

            connected_image = folderStore + '\\' + fold_name + ".avi"

            # print(i)

            array = np.array(self.get_concat_h_blank(tempImg, temp_img_plot))
            video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

        video.release()

        reg_vid = []

        save_image = folderStore + '\\' + fold_name + "_" + str(i) + "*.jpeg"

    def get_concat_h_blank(self, im1, im2, color=(0, 0, 0)):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    def check_fps(self):

        print("text is ", self.save_video_fps.get())

        if self.validate_video_speed(self.save_video_fps.get()) is True:

            try:
                self.regular_video_fps = int(self.save_video_fps.get())
                # print(set_speed)
            except ValueError:
                self.regular_video_fps = 30  # set as default speed

    def pauseRun(self):

        self.play = False

    def setValues(self, dummy=False):

        self.play = True
        self.playVideo()
        # time.sleep(1)
        # print(scene_name)

        if self.scene[self.scene_index] != self.defScene.get():
            self.img_all = np.load(self.defScene.get() + '_imgs_' + str(self.downscale_ratio) + '.npy')
            self.img_mean_list = np.load(self.defScene.get() + '_img_mean_' + str(self.downscale_ratio) + '.npy') / (2 ** self.bit_depth - 1)
            self.scene_index = self.scene.index(self.defScene.get())
            self.img_mertens = np.load(self.scene[self.scene_index] + '_mertens_imgs_' + str(self.downscale_ratio) + '.npy')

            self.resetValues()

    def playVideo(self):
        # global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play, video_speed, useMertens

        #

        if self.validate_video_speed(self.video_speed.get()) is True:

            try:
                set_speed = int(self.video_speed.get())
                # print(set_speed)
            except ValueError:
                set_speed = 50  # set as default speed

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

    def resetValues(self):
        # global verSlider, horSlider, photo, img, scene_index, play, useMertens

        self.useMertens = False
        # print("Reset")
        self.play = False
        # verSlider.config(to=stack_size[scene_index]-1)
        self.horSlider.config(to=self.frame_num[self.scene_index] - 1)
        # verSlider.set(0),
        self.horSlider.set(0)
        tempImg = Image.fromarray(self.img_all[0])
        photo = ImageTk.PhotoImage(tempImg)
        self.imagePrevlabel.configure(image=photo)
        self.updatePlot()

    def updatePlot(self):
        # global verSlider, horSlider, photo, photo_2, stack_size, img_all, img, img_mean_list, scene_index, fig

        if self.check == True:
            self.temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + int(self.verSlider.get())
        else:
            pass

        self.check == True
        # Image mean plot
        plt.close(self.fig)
        self.fig.clear()
        self.fig = plt.figure(figsize=(4, 4))  # 4.6, 3.6
        plt.plot(np.arange(self.stack_size[self.scene_index]), self.img_mean_list[(self.temp_img_ind // self.stack_size[self.scene_index]) * self.stack_size[self.scene_index]:(self.temp_img_ind //self.stack_size[self.scene_index]) *self.stack_size[self.scene_index] + self.stack_size[self.scene_index]],color='green', linewidth=2)
        plt.plot(int(self.verSlider.get()), self.img_mean_list[self.temp_img_ind], color='red', marker='o', markersize=12)
        plt.text(int(self.verSlider.get()), self.img_mean_list[self.temp_img_ind],
                 '(' + str(int(self.verSlider.get())) + ', ' + str("%.2f" % self.img_mean_list[self.temp_img_ind]) + ')', color='red',
                 fontsize=13, position=(self.verSlider.get() - 0.2, self.img_mean_list[self.temp_img_ind] + 0.04))
        plt.title('Exposure stack mean')
        plt.xlabel('Image index')
        plt.ylabel('Mean value')
        plt.xlim(-0.2, self.stack_size[self.scene_index] - 0.8)
        if self.stack_size[self.scene_index] < 20:
            plt.xticks(np.arange(0, self.stack_size[self.scene_index], 1))
        elif self.stack_size[self.scene_index] >= 15 and self.stack_size[self.scene_index] < 30:
            plt.xticks(np.arange(0, self.stack_size[self.scene_index], 2))
        else:
            plt.xticks(np.arange(0, self.stack_size[self.scene_index], 3))
        plt.ylim(-0.02, 1.1)
        plt.yticks(np.arange(0, 1.1, 0.1))
        self.fig.canvas.draw()

        self.tempImg_2 = Image.frombytes('RGB', self.fig.canvas.get_width_height(), self.fig.canvas.tostring_rgb())
        self.photo_2 = ImageTk.PhotoImage(self.tempImg_2)
        self.imagePrevlabel_2.configure(image=self.photo_2)

    def updateSlider(self, scale_value):

        # global verSlider, horSlider, photo, photo_2, imagePrevlabel, imagePrevlabel_2, img_all, img, img_mean_list, scene_index, fig, useMertens, mertensVideo
        temp_img_ind = int(self.horSlider.get()) * self.stack_size[self.scene_index] + int(self.verSlider.get())

        if self.useMertens:
            # img = self.mertensVideo[self.horSlider.get()]
            img = self.img_mertens[self.horSlider.get()]
        else:
            img = deepcopy(self.img_all[temp_img_ind])

        # self.photo = ImageTk.PhotoImage(Image.fromarray(self.img))
        # self.imagePrevlabel = tk.Label(root, image=self.photo)
        # self.imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)

        tempImg = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(tempImg)
        self.imagePrevlabel.configure(image=self.photo)

        self.updatePlot()

b = Browser(root)

root.mainloop()