from copy import deepcopy
from PIL import Image, ImageTk
import tkinter as tk


def current_frame(b):
    result = b.img_all[b.horSlider.get()]
    return result


def try1():
    print(1)


def scale_labels(b):
    text_ = b.SCALE_LABELS_NEW[0]
    tk.Label(b.root, text=text_, font=("Times New Roman", 15)).grid(row=27, column=0, )
    #temp_img_ind = int(b.horSlider.get()) * b.stack_size[0] + int(b.verSlider.get())
    updateHorSlider(b)


def updateSlider(b):
    if ((b.current_auto_exposure != "None") and (len(b.eV) > 0)):
        print(len(b.eV))
        b.verSlider.set(b.eV[b.horSlider.get()])
        temp_img_ind = int(b.horSlider.get()) * b.stack_size[b.scene_index] + b.eV[b.horSlider.get()]
    else:
        temp_img_ind = int(b.horSlider.get()) * b.stack_size[0] + int(b.verSlider.get())

    updateHorSlider(b)


def updateHorSlider(b):
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
    #image_mean_plot()



