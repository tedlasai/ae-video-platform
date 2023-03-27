def try_(arg):
    arg[0].play = True
    print(arg[0].play)
    print(arg[1])

def runVideo(args):
    self = args[0]
    self.play = True
    playVideo(args)


def validate_video_speed(speed):
    # print("text is ", video_speed)
    try:
        if int(speed):
            return True
        else:
            return False
    except ValueError:
        return True


def playVideo(args):
    # global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play, video_speed, useMertens
    self = args[0]
    root = args[1]
    if validate_video_speed(self.video_speed.get()) is True:
        try:
            set_speed = int(self.video_speed.get())
            # print(set_speed)
        except ValueError:
            set_speed = 360  # set as default speed

    # print('screen index is ', scene_index)

    if (self.horSlider.get() < (self.frame_num[self.scene_index] - 1) and self.play):
        self.horSlider.set(self.horSlider.get() + 1)
        # print("HELLO", horSlider.get())

        root.after(set_speed, playVideo(args))

    if (self.play is False):
        print("VIDEO PAUSED")

def pauseRun(self):
    self.play = False