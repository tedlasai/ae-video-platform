import browser_inputs_builder
import button_functions
import update_visulization
import manual_semantic_functions

import tkinter as tk

root = tk.Tk()
root.geometry('1400x1000'), root.title('Data Browser')  # 1900x1000+5+5


def scale_labels(value):
    """
    update the visualization of the scale label based on the index (value).
    """
    text_ = b.SCALE_LABELS_NEW[int(value)]
    tk.Label(b.root, text=text_, font=("Times New Roman", 15)).grid(row=27, column=0)
    updateHorSlider()


def updateSlider(scale_value):
    """
    update the visualization (images and plots) while the sliders were adjusted .
    """
    if ((b.current_auto_exposure != "None") and (len(b.eV) > 0)):
        print(len(b.eV))
        b.verSlider.set(b.eV[b.horSlider.get()])
    updateHorSlider()


def updateHorSlider():
    """
    update the visualization (images and plots) while the horizontal slider was adjusted .
    """

    update_visulization.updateHorSlider(b, "")
    update_visulization.updatePlot(b)


b = browser_inputs_builder.BrowserWithInputs(root)
b.init_functions()
b.buttons_builder('Pause', button_functions.pauseRun, 30, 2, para=b)
b.buttons_builder('Run', button_functions.runVideo, 29, 2, para=b)
b.buttons_builder('Reset', button_functions.resetValues, 29, 3, para=b)
b.buttons_builder('Clear Rectangles', button_functions.clear_rects, 32, 3, para=b)
b.buttons_builder('Save Interested Area', button_functions.save_interested_moving_objects_fuction, 32, 2, para=b)
b.buttons_builder('Video', button_functions.export_video, 30, 3, para=b)


def regular_video_button(self):
    self.VideoButton = tk.Button(root, text='Video', fg='#ffffff', bg='#999999', activebackground='#454545',
                                 relief=tk.RAISED, padx=10, pady=5,
                                 width=16, font=(self.widgetFont, self.widgetFontSize), command=self.export_video)
    self.VideoButton.grid(row=8 - 4, column=5, sticky=tk.E)


b.vertical_slider(scale_labels)
b.horizontal_slider(updateSlider)
b.canvas.bind('<ButtonPress-1>', lambda event, arg=b: manual_semantic_functions.on_button_press(event, arg))
b.canvas.bind('<B1-Motion>', lambda event, arg=b: manual_semantic_functions.on_move_press(event, arg))
b.canvas.bind('<ButtonRelease-1>', lambda event, arg=b: manual_semantic_functions.on_button_release(event, arg))
b.canvas.bind("<Button-3>", lambda event, arg=b: manual_semantic_functions.right_click(event, arg))
b.canvas.bind("<MouseWheel>", lambda event, arg=b: manual_semantic_functions.zoomer(event, arg))
b.canvas.bind("<Button-4>", lambda event, arg=b: manual_semantic_functions.zoomerP(event, arg))
b.canvas.bind("<Button-5>", lambda event, arg=b: manual_semantic_functions.zoomerM(event, arg))
root.mainloop()
