
import numpy as np
class RegionGrower:
    def __init__(self, img, seeds, thresh, p=1):
        self.img = img
        self.seeds = seeds
        self.thresh = thresh
        self.p = p
        self.height, self.width = img.shape
        self.seed_mark = np.zeros(img.shape)
        self.label = 1
        self.connects = self.connects_selection()

    def connects_selection(self):
        if self.p != 0:
            connects = [[-1, -1], [0, -1], [1, -1],
                        [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]
        else:
            connects = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        return connects

    def gray_diff(self, current_point, temp_point):
        return abs(int(self.img[current_point[0], current_point[1]]) - int(self.img[temp_point[0], temp_point[1]]))

    def fit(self):
        seed_list = self.seeds.copy()
        while(len(seed_list) > 0):
            current_pixel = seed_list.pop(0)
            self.seed_mark[current_pixel[0], current_pixel[1]] = self.label
            for i in range(8):
                tmpX = current_pixel[0] + self.connects[i][0]
                tmpY = current_pixel[1] + self.connects[i][1]
                if tmpX < 0 or tmpY < 0 or tmpX >= self.height or tmpY >= self.width:
                    continue
                grayDiff = self.gray_diff(current_pixel, [tmpX, tmpY])
                if grayDiff < self.thresh and self.seed_mark[tmpX, tmpY] == 0:
                    self.seed_mark[tmpX, tmpY] = self.label
                    seed_list.append([tmpX, tmpY])
        return self.seed_mark
