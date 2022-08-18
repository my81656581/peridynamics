import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time
import scipy.io
from matplotlib import pyplot as plt
from mayavi import mlab

# Load in geometry from matlab file. Chi is binary matrix specifying material existence
chi = scipy.io.loadmat('../geometries/voronoi.mat')['in'].astype(np.bool_)
whr = np.where(chi)
chi = chi[np.min(whr[0]):np.max(whr[0])+1, np.min(whr[1]):np.max(whr[1])+1]

# Bounding box for geometry; for 2D, the last entry specifies thickness
bbox = [[0, chi.shape[0]], [0,chi.shape[1]], [0,1]]
E = 7e8 # Youngs Modulus    
nu = 0.33 # Poisson's Ratio
hrad = 3.01 # Radius of "sphere of influence" (horizon)
ntau = 1000 # Number of timesteps over which boundary conditions are applied

NX = int(bbox[0][1]-bbox[0][0])

mat = pd.PDMaterial(E, nu)
geom = pd.PDGeometry(bbox, NX, np.reshape(chi,[-1],order='F'), hrad, dim=2)
model = pd.PDModel(mat, geom, None, None, dtype=np.float32, ntau=ntau)
bcs = pd.PDBoundaryConditions(geom)
# Add an applied-displacement boundary condition
# First argument is bounding box of condition, second is value of applied displacement,
# third is list of dimensions in which it is applied
bcs.addFixed([[0, 400], [0, 10], [0, 1]], 0, [1])
bcs.addFixed([[0, 400], [386, 396], [0, 1]], 40, [1])
model.setBCs(bcs)

model.solve(2000, tol=None) # Run simulation. Argument is # of timesteps

u,v,w = model.get_displacement()
xs = bcs.x
ys = bcs.y
zs = bcs.z
filt = geom.chi
M = 1
mlab.points3d(xs[filt]+M*u[filt], ys[filt] + M*v[filt], zs[filt] + M*w[filt], scale_factor=.8)
mlab.show()