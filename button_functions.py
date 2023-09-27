from copy import deepcopy
from PIL import Image

import numpy as np
import cv2
import os
import update_visulization


def runVideo(self):
    self.play = True
    playVideo(self)


def validate_video_speed(speed):
    try:
        if int(speed):
            return True
        else:
            return False
    except ValueError:
        return True


def playVideo(self):
    set_speed = self.video_speed
    if self.horSlider.get() < (self.frame_num[self.scene_index] - 1) and self.play:
        self.horSlider.set(self.horSlider.get() + 1)
        self.root.after(set_speed, lambda: playVideo(self))
    if self.play is False:
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


def save_interested_moving_objects_function(self):
    if self.current_auto_exposure == 'Semantic' and len(self.moving_rectids) > 0:
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        curr_frame = self.horSlider.get()
        temp = []
        for id in self.moving_rectids:
            coor = self.canvas.coords(id)
            temp.append([coor[1] / h, coor[0] / w, coor[3] / h, coor[2] / w])
        self.rects_without_grids_moving_objests[curr_frame] = temp.copy()


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
    self.setAutoExposure()
    self.play = False
    self.horSlider.config(to=self.frame_num[self.scene_index] - 1)
    self.horSlider.set(0)
    update_visulization.updatePlot(self)


def export_video(self):
    reg_vid = []
    self.mertensVideo = []
    self.mertens_pic = []
    if not len(self.eV) == 100:
        self.eV = [15]*100

    for i in range(100):
        self.temp_img_ind = int(i) * self.stack_size[self.scene_index] + self.eV[i]
        self.check = False
        img = deepcopy(self.img_all[i][self.eV[i]])
        reg_vid.append(img)
    m1 = Image.fromarray(reg_vid[0])
    sv = m1
    fold_name = self.scene[self.scene_index] + "_dng_pipeline_" + self.current_auto_exposure + "_FPS_" + str(
        self.video_fps)
    folderStore = os.path.join(os.path.dirname(__file__), 'Regular_Videos')
    os.makedirs(folderStore, exist_ok=True)
    connected_image = folderStore + self.joinPathChar + fold_name + ".avi"
    os.makedirs(folderStore, exist_ok=True)
    video = cv2.VideoWriter(connected_image, cv2.VideoWriter_fourcc('M', 'J', "P", 'G'), self.video_fps,
                            (sv.width, sv.height))
    for i in range(len(reg_vid)):
        tempImg = Image.fromarray(reg_vid[i])
        array = np.array(tempImg)
        video.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    video.release()
