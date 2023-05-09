import browser_builder
import tkinter as tk
import browser_inputs_builder
import button_functions
import set_auto_exposure
import update_visulization
import constants
from copy import deepcopy
from PIL import Image, ImageTk
import manual_semantic_functions

root = tk.Tk()
root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5


def scale_labels(value):
    text_ = b.SCALE_LABELS_NEW[int(value)]
    tk.Label(b.root, text=text_, font=("Times New Roman", 15)).grid(row=27, column=0)
    updateHorSlider()


def updateSlider(scale_value):
    if ((b.current_auto_exposure != "None") and (len(b.eV) > 0)):
        print(len(b.eV))
        b.verSlider.set(b.eV[b.horSlider.get()])

    updateHorSlider()


def updateHorSlider():
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

b = browser_inputs_builder.Broswer_with_inputs(root)
#b = browser_builder.Browser(root)
b.init_functions()
# b.scene_select(set_auto_exposure.setValues,para=b)
# b.auto_exposure_select(set_auto_exposure.setAutoExposure,para=b)
#b.canvas.bind('<Button-1>', lambda:button_functions.canvas_click(b,event=b.canvas.event))
b.buttons_builder('Pause',button_functions.pauseRun,1,5,para=b)
b.buttons_builder('Run',button_functions.runVideo,2,5,para=b)
b.buttons_builder('Reset',button_functions.resetValues,3,5,para=b)
b.buttons_builder('Clear Rectangles',button_functions.clear_rects,5,5,para=b)
b.buttons_builder('Save Interested Area',button_functions.save_interested_moving_objects_fuction,4,5,para=b)


b.vertical_slider(scale_labels)
b.horizontal_slider(updateSlider)
b.canvas.bind('<ButtonPress-1>', lambda event,arg=b:manual_semantic_functions.on_button_press(event,arg))
b.canvas.bind('<B1-Motion>', lambda event,arg=b:manual_semantic_functions.on_move_press(event,arg))
b.canvas.bind('<ButtonRelease-1>', lambda event,arg=b:manual_semantic_functions.on_button_release(event,arg))
b.canvas.bind("<Button-3>", lambda event,arg=b:manual_semantic_functions.right_click(event,arg))
b.canvas.bind("<MouseWheel>", lambda event,arg=b:manual_semantic_functions.zoomer(event,arg))
b.canvas.bind("<Button-4>", lambda event,arg=b:manual_semantic_functions.zoomerP(event,arg))
b.canvas.bind("<Button-5>", lambda event,arg=b:manual_semantic_functions.zoomerM(event,arg))
# b.vertical_slider(update_visulization.scale_labels,para=b)
# b.horizontal_slider(update_visulization.updateSlider,para=b)
print(b.imgSize)
root.mainloop()