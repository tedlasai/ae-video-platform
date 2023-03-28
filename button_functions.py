

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