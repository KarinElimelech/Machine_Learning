
#if you want to compute the loss
#import numpy as np

'''
class cluster - 
hold info about centroid
the rgb, and sum the pix
'''
class Cluster:
    def __init__(self, rgb):
        self.rgb = rgb
        #pixels that match the centroid
        self.pixels = 0
        self.sum =[0,0,0]
        self.pixels_to_compute_loss = []

    #add new pix to rgb sum
    def add_rgb (self, new_rgb):
        for i in range (0,3):
            self.sum[i] += new_rgb[i]

    def add_pixel (self):
        self.pixels+=1

    #update the avg of rgb-
    #happend each iteration.
    def update (self):
        new_rgb = []
        if self.pixels == 0:
             self.pixels+=1
        else :
            for i in range (0,3):
                new_rgb.append(self.sum[i] / self.pixels)
            self.rgb = new_rgb
            self.sum = [0,0,0]

    def clear_pix (self):
        self.pixels = 0

    def getRGB (self):
        return self.rgb

    '''
    #if you want to compute the loss
    def add_to_loss(self, pix):
        self.pixels_to_compute_loss.append(pix)

    def clear_loss(self):
        self.pixels_to_compute_loss.clear()

    def sum_loss(self):
        sum = 0
        for p in self.pixels_to_compute_loss:
            sum = sum + pow(np.linalg.norm(p - self.rgb),2)
        return sum
    '''
