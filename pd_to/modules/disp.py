from mayavi import mlab
import numpy as np
import time

from matplotlib import pyplot as plt
import matplotlib.animation as anim

class Display:

    def __init__(self, geom, pd, num=100):
        v0 = np.zeros((geom.NX, geom.NY, geom.NZ))
        v0[0] = 1
        fig = mlab.figure(size=(600,600))
        self.vox = mlab.pipeline.volume(mlab.pipeline.scalar_field(v0), vmin=0, vmax=1)
        self.pd = pd
        self.geom = geom
        self.num = num

    def launch_pyplot(self, geom, pd, num=100):
        NN = geom.NN
        NX = geom.NX
        NY = geom.NY
        L = geom.L
        inds = np.arange(NN)
        i = (inds%NX)
        j = (inds%(NX*NY)//NX)
        k = (inds//(NX*NY))
        xs = L*i + L/2
        ys = L*j + L/2
        zs = L*k + L/2
        fspc = 4
        M = 1
        filt = np.ones(NN)>0#(i&fspc==0) & (j%fspc==0) & (k%fspc==0)#
        tt = 0
        u = np.empty(3*NN,dtype=np.float64)
        def update_graph(n):
            pd.solve(num)
            print("Done")
            # input(";;;")
            u,v,w = pd.get_displacement()
            graph._offsets3d = (xs[filt]+M*u[filt], ys[filt]+M*v[filt], zs[filt]+M*w[filt])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        graph = ax.scatter(xs[filt], ys[filt], zs[filt])
        ani = anim.FuncAnimation(fig, update_graph)
        plt.show()

    @mlab.show
    @mlab.animate(delay=10)
    def anim(self, vox):
        while True:
            self.pd.solve(self.num)
            # u,v,w = self.pd.get_displacement()
            # vals = np.sqrt(u**2 + v**2 + w**2)
            vals = self.pd.get_fill()
            # vals = vals/np.max(vals)
            vox.mlab_source.scalars = np.reshape(vals,(self.geom.NZ,self.geom.NY,self.geom.NX)).T/np.max(vals)
            yield
    
    def launch(self):
        self.anim(self.vox)