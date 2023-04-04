import update_visulization
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

    if validate_video_speed(self.video_speed.get()) is True:
        try:
            set_speed = int(self.video_speed.get())
            # print(set_speed)
        except ValueError:
            set_speed = 360  # set as default speed
    else:
        set_speed = 360
    # print('screen index is ', scene_index)

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
    set_auto_exposure.setAutoExposure(self)
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
