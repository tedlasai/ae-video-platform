
folders = "E:\Final"  # link to directory containing all the dataset image folders


srgb_img_folder_name = 'Image_Arrays_srgb_imgs'
srgb_npy_file_name = '_show_dng_imgs.npy'
raw_img_folder_name = 'Image_Arrays_raw_imgs'
raw_npy_file_name = '_ds_raw_imgs.npy'
saliency_maps_folder_name = 'saliency_maps'
saliency_maps_npy_file_name = "_salient_maps_mbd.npy"

# os.path.join(os.path.dirname(__file__), constants.srgb_img_folder_name) + self.joinPathChar + self.scene[
#     self.scene_index] + constants.srgb_npy_file_name)

# self.raw_ims = np.load(
# os.path.join(os.path.dirname(__file__), constants.raw_img_folder_name) + self.joinPathChar + self.scene[
#     self.scene_index] + constants.raw_npy_file_nam)

# salient_map = np.load(
#     os.path.join(os.path.dirname(__file__), constants.saliency_maps_folder_name) + self.joinPathChar + self.scene[
#         self.scene_index] + constants.saliency_maps_npy_file_name)


widgetFont = 'Arial'
widgetFontSize = 12

single_stack_size = 40
single_frame_num = 100
scene = ['Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
              'Scene7', 'Scene8', 'Scene9']
frame_num = [100, 100, 100, 100, 100, 100, 100, 100, 100]  # number of frames per position
stack_size = [40, 40, 40, 40, 40, 40, 40, 40, 40]  # number of shutter options per position
num_hist_bins = 100
SCALE_LABELS = {
    0: '15"',
    1: '8"',
    2: '6"',
    3: '4"',
    4: '2"',
    5: '1"',
    6: '0"5',
    7: '1/4',
    8: '1/8',
    9: '1/15',
    10: '1/30',
    11: '1/60',
    12: '1/125',
    13: '1/250',
    14: '1/500'
}
SCALE_LABELS_NEW = {
    0: '15"',
    1: '13"',
    2: '10"',
    3: '8"',
    4: '6"',
    5: '5"',
    6: '4"',
    7: '3"2',
    8: '2"5',
    9: '2"',
    10: '1"6',
    11: '1"3',
    12: '1"',
    13: '0"8',
    14: '0"6',
    15: '0"5',
    16: '0"4',
    17: '0"3',
    18: '1/4',
    19: '1/5',
    20: '1/6',
    21: '1/8',
    22: '1/10',
    23: '1/13',
    24: '1/15',
    25: '1/20',
    26: '1/25',
    27: '1/30',
    28: '1/40',
    29: '1/50',
    30: '1/60',
    31: '1/80',
    32: '1/100',
    33: '1/125',
    34: '1/160',
    35: '1/200',
    36: '1/250',
    37: '1/320',
    38: '1/400',
    39: '1/500'
}

SCALES = [15, 8, 6, 4, 2, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 15, 1 / 30, 1 / 60, 1 / 125, 1 / 250, 1 / 500]
NEW_SCALES = [15,13,10,8,6,5,4,3.2,2.5,2,1.6,1.3,1,0.8,0.6,0.5,0.4,0.3,1/4,1/5,1/6,1/8,1/10,1/13,1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100,1/125,1/160,1/200,1/250,1/320,1/400,1/500]
indexes_out_of_40 = [0, 3, 4, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
auto_exposures = ["None", "Global","Saliency_map",'Semantic','Entropy']
# auto_exposures = ["None", "Global","Saliency_map",'Semantic','Max Gradient srgb','Entropy','HDR Histogram Method']
downscale_ratio = 0.12
imgSize = [int(4480 * downscale_ratio), int(6720 * downscale_ratio)]
video_fps = 8
video_speed = 360
num_bins = 100