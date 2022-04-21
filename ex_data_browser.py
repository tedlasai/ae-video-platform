import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mp
mp.rcParams.update({'axes.titlesize': 14, 'font.size': 11, 'font.family': 'arial'})

def setValues():
    global scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth
    if scene[scene_index] != defScene.get():
        img_all = np.load(defScene.get()+'_imgs_'+str(downscale_ratio)+'.npy')
        img_mean_list=np.load(defScene.get()+'_img_mean_'+str(downscale_ratio)+'.npy')/(2**bit_depth-1)
        scene_index=scene.index(defScene.get())
        resetValues()

def resetValues():
    global verSlider, horSlider, photo, img, scene_index
    verSlider.config(to=stack_size[scene_index]-1)
    horSlider.config(to=frame_num[scene_index]-1)
    verSlider.set(0), horSlider.set(0)
    tempImg = Image.fromarray(img_all[0])
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)
    updatePlot()

def updatePlot():
    global verSlider, horSlider, photo, photo_2, stack_size, img_all, img, img_mean_list, scene_index, fig
    temp_img_ind=int(horSlider.get())*stack_size[scene_index]+int(verSlider.get())
    
    #Image mean plot
    plt.close(fig)
    fig.clear()
    fig=plt.figure(figsize=(4.6, 3.6))
    plt.plot(np.arange(stack_size[scene_index]),img_mean_list[(temp_img_ind//stack_size[scene_index])*stack_size[scene_index]:(temp_img_ind//stack_size[scene_index])*stack_size[scene_index]+stack_size[scene_index]], color='green', linewidth=2)
    plt.plot(int(verSlider.get()),img_mean_list[temp_img_ind], color='red', marker='o', markersize=12)
    plt.text(int(verSlider.get()),img_mean_list[temp_img_ind],'('+str(int(verSlider.get()))+', '+str("%.2f" % img_mean_list[temp_img_ind])+')',color='red', fontsize=13, position=(verSlider.get()-0.2,img_mean_list[temp_img_ind]+0.04))
    plt.title('Exposure stack mean')
    plt.xlabel('Image index')
    plt.ylabel('Mean value')
    plt.xlim(-0.2, stack_size[scene_index]-0.8)
    if stack_size[scene_index] < 20:
        plt.xticks(np.arange(0,stack_size[scene_index],1))
    elif stack_size[scene_index] >= 15 and stack_size[scene_index] < 30:
        plt.xticks(np.arange(0,stack_size[scene_index],2))
    else:
        plt.xticks(np.arange(0,stack_size[scene_index],3))
    plt.ylim(-0.02, 0.85)
    plt.yticks(np.arange(0,0.85,0.1))
    fig.canvas.draw()
    
    tempImg_2 = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    photo_2 = ImageTk.PhotoImage(tempImg_2)
    imagePrevlabel_2.configure(image=photo_2)
    
def updateSlider(event):
    global verSlider, horSlider, photo, photo_2, imagePrevlabel, imagePrevlabel_2, img_all, img, img_mean_list, scene_index, fig
    temp_img_ind=int(horSlider.get())*stack_size[scene_index]+int(verSlider.get())
    img=deepcopy(img_all[temp_img_ind])
    tempImg = Image.fromarray(img)
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)
    
    updatePlot()

################################################################################Main()
widgetFont, widgetFontSize= 'Arial', 14

scene = ['Scene1','Scene2', 'Scene3']
frame_num = [90,65, 15]
stack_size = [12,47, 28]

scene_index=0

bit_depth = 8

downscale_ratio=0.12

imgSize = [int(4480 * downscale_ratio), int(6720 * downscale_ratio)]
widthToScale=imgSize[1]
widPercent = (widthToScale/float(imgSize[1]))
heightToScale = int(float(imgSize[0])*float(widPercent))

img_all = np.load(scene[scene_index]+'_imgs_'+str(downscale_ratio)+'.npy')
img_mean_list=np.load(scene[scene_index]+'_img_mean_'+str(downscale_ratio)+'.npy')/(2**bit_depth-1)

tempImg, img = deepcopy(img_all[0]), deepcopy(img_all[0])
#################################

#Tkinter Window
root=tk.Tk()
root.geometry('1900x1000+5+5'), root.title('Data Browser'), root.iconbitmap('AF_Icon.ico')

#Image Convas
tempImg = Image.fromarray(tempImg)
photo = ImageTk.PhotoImage(tempImg)
imagePrevlabel=tk.Label(root, image=photo)
imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)

#Image mean plot
fig=plt.figure(figsize=(4.6, 3.6))
plt.plot(np.arange(stack_size[scene_index]),img_mean_list[0:stack_size[scene_index]], color='green', linewidth=2)#,label='Exposure stack mean')
plt.plot(0,img_mean_list[0], color='red', marker='o', markersize=12)
plt.text(0,img_mean_list[0],'('+str(0)+', '+str("%.2f" % img_mean_list[0])+')',color='red', fontsize=13, position=(0-0.2,img_mean_list[0]+0.04))
plt.title('Exposure stack mean')
plt.xlabel('Image index')
plt.ylabel('Mean value')
plt.xlim(-0.2, stack_size[scene_index]-0.8)
if stack_size[scene_index] < 20:
    plt.xticks(np.arange(0,stack_size[scene_index],1))
elif stack_size[scene_index] >= 15 and stack_size[scene_index] < 30:
    plt.xticks(np.arange(0,stack_size[scene_index],2))
else:
    plt.xticks(np.arange(0,stack_size[scene_index],3))
plt.ylim(-0.02, 0.85)
plt.yticks(np.arange(0,0.85,0.1))
fig.canvas.draw()

tempImg_2 = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
photo_2 = ImageTk.PhotoImage(tempImg_2)
imagePrevlabel_2=tk.Label(root, image=photo_2)
imagePrevlabel_2.grid(row=4, column=2,  columnspan=2, rowspan=24, sticky=tk.NE)

#Vertical Slider
verSliderLabel=tk.Label(root, text='Exposure Time', font=(widgetFont, widgetFontSize))
verSliderLabel.grid(row=0, column=0)
verSlider=tk.Scale(root, activebackground= 'black', cursor='sb_v_double_arrow', from_=0, to=stack_size[scene_index]-1, font=(widgetFont, widgetFontSize), length=heightToScale, command=updateSlider)
verSlider.grid(row=1, column=0, rowspan=30)

#Horizantal Slider
horSlider=tk.Scale(root, activebackground= 'black', cursor='sb_h_double_arrow', from_=0, to=frame_num[0]-1, label='Frame Number', font=(widgetFont, widgetFontSize), orient=tk.HORIZONTAL, length=widthToScale, command=updateSlider)
horSlider.grid(row=31, column=1, sticky=tk.SW)

#Select Scene List
defScene = tk.StringVar(root)
defScene.set(scene[scene_index]) # default value
selSceneLabel=tk.Label(root, text='Select Scene:', font=(widgetFont, widgetFontSize))
selSceneLabel.grid(row=0, column=2, sticky=tk.W)

sceneList = tk.OptionMenu(root, defScene, *scene)
sceneList.config(font=(widgetFont, widgetFontSize-2), width=15, anchor=tk.W)
sceneList.grid(row=1, column=2, sticky=tk.NE)

#Run Button
RunButton=tk.Button(root, text='Run', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=18, font=(widgetFont, widgetFontSize), command=setValues)
RunButton.grid(row=29, column=2, sticky=tk.E)

#Reset Button
RunButton=tk.Button(root, text='Reset', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=18, font=(widgetFont, widgetFontSize), command=resetValues)
RunButton.grid(row=31, column=2, sticky=tk.E)

root.mainloop()