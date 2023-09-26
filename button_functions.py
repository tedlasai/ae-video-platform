
from copy import deepcopy
from PIL import Image

import numpy as np
import cv2
import os
import update_visulization
import regular
import high_res_auto_ex_video
import set_auto_exposure


def runVideo(self):
    self.play = True
    playVideo(self)


def validate_video_speed(speed):
    # print("text is ", video_speed)
    try:
        if int(speed):
            return True
        else:
            return False
    except ValueError:
        return True


def playVideo(self):
    # global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play, video_speed, useMertens

    # if validate_video_speed(self.video_speed) is True:
    #     try:
    #         set_speed = int(self.video_speed)
    #         # print(set_speed)
    #     except ValueError:
    #         set_speed = 360  # set as default speed
    # else:
    #     set_speed = 360
    # print('screen index is ', scene_index)
    set_speed = self.video_speed
    if (self.horSlider.get() < (self.frame_num[self.scene_index] - 1) and self.play):
        self.horSlider.set(self.horSlider.get() + 1)
        # print("HELLO", horSlider.get())

        self.root.after(set_speed, lambda: playVideo(self))

    if (self.play is False):
        print("VIDEO PAUSED")


def pauseRun(self):
    self.play = False


def clear_rects(self):
    clear_rects_local(self)
    clear_rects_local_wo_grids(self)
    clear_moving_rects(self)
    self.the_moving_area_list = []
    if self.making_a_serious_of_videos == 0:
        self.rects_without_grids_moving_objests = {}
    print("clear lenth of moving areas")
    print(len(self.the_moving_area_list))


def save_interested_moving_objects_fuction(self):
    if self.current_auto_exposure == 'Semantic' and len(self.moving_rectids) > 0:
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


def clear_moving_rects(self):
    for rect in self.moving_rectids:
        self.canvas.delete(rect)
    self.moving_rectids = []
    if self.making_a_serious_of_videos == 0:
        self.rects_without_grids_moving_objests = {}


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
    update_visulization.updatePlot(self)


def export_video(self):
    reg_vid = []
    reg_vid_plot = []
    # list = ['15', '8', '6', '4', '2', '1', '05', '1-4', '1-8', '1-15', '1-30', '1-60', '1-125', '1-250', '1-500']

    self.mertensVideo = []
    self.mertens_pic = []

    #if self.res_check == 0 and self.current_auto_exposure == "None":
    if self.current_auto_exposure == "None":
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

    # elif self.res_check == 0 and self.current_auto_exposure != "None":
    else:
        if not len(self.eV) == 100:
            print(self.eV)
            print("1")
            return

        for i in range(100):
            print("2")
            self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + self.eV[i]
            self.check = False
            #self.updatePlot()
            # reg_vid_plot.append(self.tempImg_2)

            # img = deepcopy(self.img_all[self.temp_img_ind])
            img = deepcopy(self.img_all[i][self.eV[i]])
            print("IMG", img.shape)

            reg_vid.append(img)

        m1 = Image.fromarray(reg_vid[0])
        # m2 = reg_vid_plot[0]
        # sv = self.get_concat_h_blank(m1, m2)
        sv = m1

        #self.check_fps()

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

        # self.check_fps()

    # elif self.res_check == 1 and self.current_auto_exposure == "None":
    #
    #     self.check_fps()
    #
    #     regular.main(self.scene[self.scene_index], self.video_fps, self.verSlider.get(), list, self.folders)
    #
    # elif self.res_check == 1 and self.current_auto_exposure == "Global" or self.current_auto_exposure == 'Local':
    #
    #     self.check_fps()
    #
    #     high_res_auto_ex_video.main(self.scene[self.scene_index], self.video_fps, self.eV,
    #                                 self.current_auto_exposure, self.folders)
