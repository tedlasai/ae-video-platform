
folders = "E:\Final"  # link to directory containing all the dataset image folders

widgetFont = 'Arial'
widgetFontSize = 12

scene = ['Scene1', 'Scene2', 'Scene3', 'Scene4', 'Scene5', 'Scene6',
              'Scene7', 'Scene8', 'Scene9', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14', 'Scene15',
              'Scene16', 'Scene17', 'Scene18', 'Scene19', 'Scene20', 'Scene21','Scene22']
frame_num = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                  100, 100, 100, 100, 100,100]  # number of frames per position
stack_size = [40, 15, 40, 15, 15, 40, 15, 15, 15, 40, 40, 15, 15, 15, 40, 40, 40,
                   40, 40, 40, 40, 40]  # number of shutter options per position

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
NEW_SCALES = [15,13,10,8,6,5,4,3.2,2.5,2,1.6,1.3,1,0.8,0.6,0.5,0.4,0.3,1/4,1/5,1/6,1/8,1/10,1/13,1/15,1/20,1/25,1/30,1/40,1/50,1/60,1/80,1/100,1/125,1/160,1/200,1/250,1/320,1/400,1/500]

auto_exposures = ["None", "Global","Saliency_map", "Local", 'Local without grids', 'Local on moving objects','Max Gradient srgb','Entropy','HDR Histogram Method']

