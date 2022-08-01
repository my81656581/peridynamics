from mayavi import mlab
import numpy as np
import time

from matplotlib import pyplot as plt
import matplotlib.animation as anim

class Display:

    def __init__(self, geom, pd, opt=None, num=10000, tol = 0.001):
        v0 = np.zeros((geom.NX, geom.NY, geom.NZ))
        v0[0] = 1
        self.v0 = v0
        self.pd = pd
        self.geom = geom
        self.num = num
        self.opt = opt
        self.tol = tol

    def launch_pyplot(self, num=100):
        geom = self.geom
        pd = self.pd
        NN = geom.NN
        xs = pd.bcs.x
        ys = pd.bcs.y
        zs = pd.bcs.z
        M = 1
        filt = geom.chi
        def update_graph(n):
            pd.solve(num)
            u,v,w = pd.get_displacement()
            graph._offsets3d = (xs[filt]+M*u[filt], ys[filt]+M*v[filt], zs[filt]+M*w[filt])

        u,v,w = pd.get_displacement()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        graph = ax.scatter(xs[filt]+M*u[filt], ys[filt]+M*v[filt], zs[filt]+M*w[filt])
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ani = anim.FuncAnimation(fig, update_graph)
        plt.show()

    @mlab.show
    @mlab.animate(delay=10)
    def anim(self, vox):
        while True:
            self.pd.solve(self.num, tol = self.tol)
            # print("Updating")
            self.opt.step(self.pd)
            # u,v,w = self.pd.get_displacement()
            # vals = np.sqrt(u**2 + v**2 + w**2)
            vals = self.pd.get_fill()
            vals[vals<0.8] = 0
            # vals = vals/np.max(vals)
            vox.mlab_source.scalars = np.reshape(vals,(self.geom.NZ,self.geom.NY,self.geom.NX)).T/np.max(vals)
            yield
    
    def launch(self):
        fig = mlab.figure(size=(600,600))
        vox = mlab.pipeline.volume(mlab.pipeline.scalar_field(self.v0), vmin=0, vmax=1)
        self.anim(vox)

    def launch_2D(self):
        NX, NY = self.geom.NX, self.geom.NY
        fig, ax = plt.subplots(1)
        heatmap = ax.pcolor(np.ones((NY, NX)))
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.canvas.draw()
        fig.show()

        while True:
            self.pd.solve(5000, tol = self.tol)
            self.opt.step(self.pd)
            heatmap = plt.pcolor(np.reshape(self.pd.get_fill(), (NY, NX)))
            ax.draw_artist(ax.patch)
            ax.draw_artist(heatmap)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()