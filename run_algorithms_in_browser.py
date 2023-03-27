import browser_builder
import tkinter as tk
import browser_inputs_builder
import button_functions
import constants


root = tk.Tk()
root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5


def fun1():
    print(b.play)

def fun2():
    print(2)



b = browser_inputs_builder.Broswer_with_inputs(root)
#b = browser_builder.Browser(root)
b.init_functions()

b.buttons_builder("Pause",button_functions.pauseRun,1,5,para=b)
b.buttons_builder("Run",button_functions.runVideo,2,5,para=[b,root])

print(b.imgSize)
root.mainloop()