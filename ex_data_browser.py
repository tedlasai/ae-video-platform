import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mp

mp.rcParams.update({'axes.titlesize': 14, 'font.size': 11, 'font.family': 'arial'})

def HdrMean():
    global horSlider, photo, imagePrevlabel, img_all, scene_index
    temp_img_ind=int(horSlider.get()*stack_size[scene_index])
    temp_stack=deepcopy(img_all[temp_img_ind:temp_img_ind+stack_size[scene_index]])

    temp_img=(np.mean(temp_stack, axis=0)).astype(np.uint8)
    cv2.putText(temp_img, 'HDR-Mean', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    tempImg = Image.fromarray(temp_img)
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)
    
def HdrMedian():
    global horSlider, photo, imagePrevlabel, img_all, scene_index
    temp_img_ind=int(horSlider.get()*stack_size[scene_index])
    temp_stack=deepcopy(img_all[temp_img_ind:temp_img_ind+stack_size[scene_index]])
    
    temp_img=(np.median(temp_stack, axis=0)).astype(np.uint8)
    cv2.putText(temp_img, 'HDR-Median', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    tempImg = Image.fromarray(temp_img)
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)
    
def HdrMertens():
    global horSlider, photo, imagePrevlabel, img_all, scene_index
    temp_img_ind=int(horSlider.get()*stack_size[scene_index])
    temp_stack=deepcopy(img_all[temp_img_ind:temp_img_ind+stack_size[scene_index]])
    
    # Exposure fusion using Mertens
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(temp_stack)
    # Convert datatype to 8-bit and save
    res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
    cv2.putText(res_mertens_8bit, 'HDR-Mertens', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    tempImg = Image.fromarray(res_mertens_8bit)
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)
    
def HdrAbdullah():
    global horSlider, photo, imagePrevlabel, img_all, scene_index
    temp_img_ind=int(horSlider.get()*stack_size[scene_index])
    temp_stack=deepcopy(img_all[temp_img_ind:temp_img_ind+stack_size[scene_index]])
    
    print(temp_img_ind,temp_img_ind+stack_size[scene_index])
    
    temp_stack_clip_weights=np.where(temp_stack[:,:,:,1] < 10, 0.1, 1)
    mean_2=np.average(temp_stack[:,:,:,1], axis=0, weights=temp_stack_clip_weights)
    stack_distance=abs(temp_stack[:,:,:,1]-mean_2)
    stack_distance_arrs=np.argsort(stack_distance, axis=0)
    stack_distance_min=(np.mean(stack_distance_arrs[0:7], axis=0)).astype(np.uint8)
    stack_distance_min_med = cv2.medianBlur(stack_distance_min,91)
    stack_distance_min_med = cv2.medianBlur(stack_distance_min_med,151)
    stack_distance_min_med = cv2.medianBlur(stack_distance_min_med,191)
    mean_min_dis_med=(np.zeros((temp_stack.shape[1],temp_stack.shape[2],temp_stack.shape[3]))).astype(np.uint8)
    for i in range (mean_min_dis_med.shape[0]):
        for j in range (mean_min_dis_med.shape[1]):
            mean_min_dis_med[i,j,:]=temp_stack[stack_distance_min_med[i,j],i,j,:]
    cv2.putText(mean_min_dis_med, 'HDR-Abdullah', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    tempImg = Image.fromarray(mean_min_dis_med)
    photo = ImageTk.PhotoImage(tempImg)
    imagePrevlabel.configure(image=photo)

def setValues(args = None):
    global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play

    play = True
    playVideo()
    #time.sleep(1)


    if scene[scene_index] != defScene.get():
        img_all = np.load(defScene.get() + '_imgs_' + str(downscale_ratio) + '.npy')
        img_mean_list = np.load(defScene.get() + '_img_mean_' + str(downscale_ratio) + '.npy')/(2**bit_depth-1)
        scene_index = scene.index(defScene.get())
        resetValues()



def playVideo():

    global horSlider, scene, scene_index, defScene, img, img_all, img_mean_list, downscale_ratio, bit_depth, play, video_speed

    if validate_video_speed(video_speed.get()) is True:

        try:
            set_speed = int(video_speed.get())
            # print(set_speed)
        except ValueError:
            set_speed = 50 # set as default speed

    # print('screen index is ', scene_index)
    if (horSlider.get() < (frame_num[scene_index]-1) and play):

        horSlider.set(horSlider.get() + 1)
        # print("HELLO", horSlider.get())
        root.after(set_speed, playVideo)

    if (play is False):

        print("VIDEO PAUSED")


def validate_video_speed(video_speed):
    # print("text is ", video_speed)
    try:
        if int(video_speed):
            return True
        else:
            return False
    except ValueError:
        return True

def resetValues():

    global verSlider, horSlider, photo, img, scene_index, play
    # print("Reset")
    play = False
    # verSlider.config(to=stack_size[scene_index]-1)
    horSlider.config(to=frame_num[scene_index]-1)
    # verSlider.set(0),
    horSlider.set(0)
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
    fig=plt.figure(figsize=(4, 4)) #4.6, 3.6
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
widgetFont, widgetFontSize= 'Arial', 12

scene = ['Scene101','Scene102', 'Scene103','Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15', 'Scene16', 'Scene17']
frame_num = [90,65, 15, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100] # number of frames per position
stack_size = [12,47, 28, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15] # number of shutter options per position

scene_index=0

bit_depth = 8

downscale_ratio=0.12

imgSize = [int(4480 * downscale_ratio), int(6720 * downscale_ratio)]
widthToScale=imgSize[1]
widPercent = (widthToScale/float(imgSize[1]))
heightToScale = int(float(imgSize[0])*float(widPercent))

img_all = np.load(scene[scene_index]+'_imgs_'+str(downscale_ratio)+'.npy')
img_mean_list=np.load(scene[scene_index]+'_img_mean_'+str(downscale_ratio)+'.npy')/(2**bit_depth-1)

img = deepcopy(img_all[0])
#################################

#Tkinter Window
root=tk.Tk()
root.geometry('1600x900'), root.title('Data Browser'), root.iconbitmap('AF_Icon.ico') #1900x1000+5+5

#Image Convas
photo = ImageTk.PhotoImage(Image.fromarray(img))
imagePrevlabel=tk.Label(root, image=photo)
imagePrevlabel.grid(row=1, column=1, rowspan=30, sticky=tk.NW)

#TextBox
video_speed = tk.StringVar()
# video_speed = 1
tk.Label(root, text="Playback Speed").grid(row=34, column=1)
e1 = tk.Entry(root, textvariable = video_speed)
e1.grid(row=35, column=1)

#Image mean plot
fig=plt.figure(figsize=(4, 4)) # 4.6, 3.6
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
sceneList = tk.OptionMenu(root, defScene, *scene, command=setValues)
sceneList.config(font=(widgetFont, widgetFontSize-2), width=15, anchor=tk.W)
sceneList.grid(row=1, column=2, sticky=tk.NE)

#HDR Button - Mean
HdrMeanButton=tk.Button(root, text='HDR-Mean', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=16, font=(widgetFont, widgetFontSize), command=HdrMean)
HdrMeanButton.grid(row=31, column=2, sticky=tk.E) #initial row was 26, +1 increments for all other rows

#HDR Button - Median
HdrMedianButton=tk.Button(root, text='HDR-Median', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=16, font=(widgetFont, widgetFontSize), command=HdrMedian)
HdrMedianButton.grid(row=32, column=2, sticky=tk.E)

#HDR Button - Mertens
HdrMertensButton=tk.Button(root, text='HDR-Mertens', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=16, font=(widgetFont, widgetFontSize), command=HdrMertens)
HdrMertensButton.grid(row=33, column=2, sticky=tk.E)

#HDR Button - Abdullah
HdrAbdullahButton=tk.Button(root, text='HDR-Abdullah', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=16, font=(widgetFont, widgetFontSize), command=HdrAbdullah)
HdrAbdullahButton.grid(row=34, column=2, sticky=tk.E)


#Run Button
RunButton=tk.Button(root, text='Run', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=16, font=(widgetFont, widgetFontSize), command=setValues)
RunButton.grid(row=35, column=2, sticky=tk.E)

#Reset Button
RestButton=tk.Button(root, text='Reset', fg='#ffffff', bg='#999999', activebackground='#454545', relief=tk.RAISED, width=16, font=(widgetFont, widgetFontSize), command=resetValues)
RestButton.grid(row=36, column=2, sticky=tk.E)



root.mainloop()



