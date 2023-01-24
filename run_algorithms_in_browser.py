import broswer_builder
import tkinter as tk
import constants


root = tk.Tk()
root.geometry('1600x900'), root.title('Data Browser')  # 1900x1000+5+5


def fun1():
    print(1)

def fun2():
    print(2)

b = broswer_builder.Browser(root)
b.buttons_builder(root,"xx",fun1,5,5)
b.buttons_builder(root,"yy",fun2,7,6)

print(b.imgSize)
root.mainloop()