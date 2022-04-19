from mayavi import mlab
import numpy as np
import time

class Display:

    def __init__(self, geom, pd):
        v0 = np.zeros((geom.NX, geom.NY, geom.NZ))
        v0[0] = 1
        fig = mlab.figure(size=(600,600))
        self.vox = mlab.pipeline.volume(mlab.pipeline.scalar_field(v0), vmin=0, vmax=1)
        self.pd = pd
        self.geom = geom

    @mlab.show
    @mlab.animate(delay=10)
    def anim(self, vox):
        while True:
            self.pd.solve(50)
            u,v,w = self.pd.get_displacement()
            vals = np.abs(u)
            vals = vals/np.max(vals)
            vox.mlab_source.scalars = np.reshape(vals,(self.geom.NZ,self.geom.NY,self.geom.NX)).T/np.max(vals)
            yield
    
    def launch(self):
        self.anim(self.vox)