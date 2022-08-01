import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time
from mayavi import mlab

# 3D Cantilever, bottom edge loaded
bbox = [[0, 2], [0, 1], [0, 1]]
NX = 80 # 80, 90, 100
E = 1
nu = 0.25
hrad = 3.01
rho = 0.1
ntau = 500
volfrac = 0.3
penal = 3
alpha = 0.7
mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, None, hrad, dim=3)
bcs = pd.PDBoundaryConditions(geom)
bcs.addFixed([[0, .1], [0, 1], [0, 1]], 0, [0,1,2])
# bcs.addFixed([[1.95,2], [0, .05], [0, 1]], -.5, [1])
bcs.addDistributedForce([[1.95,2], [0, .05], [0, 1]], [0,-0.001,0])
opt = pd.PDOptimizer(alpha, volfrac, penal)
model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float32,ntau=ntau)
for _ in range(50):
    model.solve(10000, tol=0.01, minT = 100)
    opt.step(model)
vals = model.get_fill()
vals[vals<0.8] = 0
vals = np.reshape(vals,(geom.NZ,geom.NY,geom.NX)).T/np.max(vals)
fig = mlab.figure(size=(600,600))
vox = mlab.pipeline.volume(mlab.pipeline.scalar_field(vals), vmin=0, vmax=1)
mlab.show()
