import numpy as np
import exposure_class


num_frame = 100
num_imgs_per_frame = 15
def local_interested_grids_generater(y_num_grids,x_num_grid,list_of_grid_coordinates):
    interested_grids = np.zeros((y_num_grids,x_num_grid))
    for item in list_of_grid_coordinates:
        interested_grids[item[0]][item[1]] = 1
    frame_interested_grids = np.repeat([interested_grids],num_imgs_per_frame,axis=0)
    scene_intereted_grids = np.repeat([frame_interested_grids],num_frame,axis=0)
    result = np.where(scene_intereted_grids == 1)
    list_local=list(zip(result[0], result[1],result[2],result[3]))
    return list_local
