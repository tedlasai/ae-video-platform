def draw_rectangle(self, row, col, color):
    ww = self.photo.width()
    hh = self.photo.height()
    topx = col * (ww // self.col_num_grids)
    if col == self.col_num_grids - 1:
        botx = ww - 1
    else:
        botx = (col + 1) * (ww // self.col_num_grids)
    topy = row * (hh // self.row_num_grids)
    if row == self.row_num_grids - 1:
        boty = hh - 1
    else:
        boty = (row + 1) * (hh // self.row_num_grids)
    rect = self.canvas.create_rectangle(topx, topy, botx, boty, fill='', outline=color)
    return rect


def local_wo_grids(event, self):
    self.on_button_press(event, self)
    self.on_move_press(event, self)
    self.on_button_release(event, self)


def on_button_press(event, self):
    self.start_x = self.canvas.canvasx(event.x)
    self.start_y = self.canvas.canvasy(event.y)
    if self.current_auto_exposure == "Semantic":
        self.curX = self.start_x
        self.curY = self.start_y

        for i, r in enumerate(self.moving_rectids):

            r_start_x, r_start_y, r_end_x, r_end_y = self.canvas.coords(r)
            if r_start_x <= self.start_x <= r_end_x and r_start_y <= self.start_y <= r_end_y:
                self.the_moving_rect = r
                self.rect_ind = i
                break
        # create rectangle if not yet exist
        if not self.the_moving_rect:
            rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, outline='indigo')
            self.moving_rectids.append(rect)


def on_move_press(event, self):
    if self.current_auto_exposure == "Semantic":
        curX = self.canvas.canvasx(event.x)
        curY = self.canvas.canvasy(event.y)

        # expand rectangle as you drag the mouse
        if self.the_moving_rect is None:
            self.canvas.coords(self.moving_rectids[-1], self.start_x, self.start_y, curX, curY)
        else:
            self.x_offset = curX - self.curX
            self.y_offset = curY - self.curY
            self.canvas.move(self.the_moving_rect, self.x_offset, self.y_offset)
        self.curX = curX
        self.curY = curY


def on_button_release(event, self):
    print("rect: " + str(self.rect))
    print("start_x: " + str(self.start_x))
    print("start_y: " + str(self.start_y))
    print("cur_x: " + str(self.curX))
    print("cur_y: " + str(self.curY))
    if self.current_auto_exposure == "Semantic":
        if self.the_moving_rect is not None:
            self.the_moving_rect = None
            self.x_offset = 0
            self.y_offset = 0
            self.rect_ind = None


def right_click(event, self):
    self.start_x = self.canvas.canvasx(event.x)
    self.start_y = self.canvas.canvasy(event.y)
    self.curX = self.start_x
    self.curY = self.start_y
    for i, r in enumerate(self.moving_rectids):
        r_start_x, r_start_y, r_end_x, r_end_y = self.canvas.coords(r)
        if r_start_x <= self.start_x <= r_end_x and r_start_y <= self.start_y <= r_end_y:
            self.the_scrolling_rect = r
            break


def zoomerP(event, self):
    if self.the_scrolling_rect:
        old_coordinate = self.canvas.coords(self.the_scrolling_rect)
        factor1 = 0.9
        factor2 = 1.1
        self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * factor1, old_coordinate[1] * factor1,
                           old_coordinate[2] * factor2, old_coordinate[3] * factor2)


def zoomerM(event, self):
    print(self.the_scrolling_rect)
    if self.the_scrolling_rect:
        old_coordinate = self.canvas.coords(self.the_scrolling_rect)
        print(old_coordinate)
        factor1 = 0.9
        factor2 = 1.1
        self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * factor2, old_coordinate[1] * factor2,
                           old_coordinate[2] * factor1, old_coordinate[3] * factor1)


def zoomer(event, self):
    if self.the_scrolling_rect:
        factor1 = 0.9
        factor2 = 1.1
        if (event.delta > 0):
            old_coordinate = self.canvas.coords(self.the_scrolling_rect)
            self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * factor1, old_coordinate[1] * factor1,
                               old_coordinate[2] * factor2, old_coordinate[3] * factor2)
        elif (event.delta < 0):
            old_coordinate = self.canvas.coords(self.the_scrolling_rect)
            print(old_coordinate)
            self.canvas.coords(self.the_scrolling_rect, old_coordinate[0] * factor2, old_coordinate[1] * factor2,
                               old_coordinate[2] * factor1, old_coordinate[3] * factor1)


def list_local_without_grids_moving_objects(self):
    list_ = []
    keys = self.rects_without_grids_moving_objests.keys()
    keys = sorted(keys)
    if len(keys) > 0:
        list_coordi_temp = self.rects_without_grids_moving_objests[keys[0]]
        for i in range(keys[0] + 1):
            list_.append(list_coordi_temp.copy())
        for i in range(1, len(keys)):
            list_pre = list_coordi_temp.copy()
            list_coordi_temp = self.rects_without_grids_moving_objests[keys[i]]
            gap = keys[i] - keys[i - 1]
            for j in range(1, gap):
                # assume the number of rects are the same. if not, follow the less one, and assume the first "size" of rects are the cooresponding ones
                size = min(len(list_pre), len(list_coordi_temp))
                list_coordi_temp_gap = []
                for k in range(size):
                    a1, b1, c1, d1 = list_pre[k]
                    a2, b2, c2, d2 = list_coordi_temp[k]
                    a = a1 + (a2 - a1) * j / gap
                    b = b1 + (b2 - b1) * j / gap
                    c = c1 + (c2 - c1) * j / gap
                    d = d1 + (d2 - d1) * j / gap
                    list_coordi_temp_gap.append([a, b, c, d])
                list_.append(list_coordi_temp_gap.copy())
            list_.append(list_coordi_temp.copy())
        for i in range(keys[-1] + 1, self.frame_num[self.scene_index]):
            list_.append(list_coordi_temp.copy())
    self.the_moving_area_list = list_.copy()
    return list_
