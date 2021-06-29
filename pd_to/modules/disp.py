
import numpy as np
from matplotlib import pyplot as plt


class Disp:
    def __init__(self, geom):
        coords = geom.d_coords.copy_to_host()
        x,y = np.unique(coords[0,:]),np.unique(coords[1,:])
        X,Y = np.meshgrid(x,y)
        self.NY, self.NX = X.shape
        self.fig, self.ax = plt.subplots(1)
        data = np.ones((self.NY, self.NX))
        heatmap = self.ax.pcolor(data)
        self.fig.canvas.draw()
        self.fig.show()

    def update(self, d_k):
        data = np.reshape(d_k.copy_to_host(), (self.NY, self.NX))
        heatmap = self.ax.pcolor(data)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(heatmap)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()