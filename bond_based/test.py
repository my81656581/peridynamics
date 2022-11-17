import numpy as np
import modules.pd as pd
from mayavi import mlab
from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

NX = 50
hrad = 2.01
E = 7e8
nu = 0.25
name = "Ankle Exo Left Reduced"

mat = pd.PDMaterial(E, nu)
geom = pd.PDGeometry(NX, hrad = hrad, filename=name)
bcs = pd.PDBoundaryConditions(geom)
model = pd.PDModel(mat, geom, bcs, dtype=np.float32, ntau=5000)

 
# Create a new plot
# figure = plt.figure()
# axes = mplot3d.Axes3D(figure)
# your_mesh = mesh.Mesh.from_file('../geometries/' + name + '.stl')
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors*geom.sclfac, alpha=0.5, edgecolor='k'))
# scale = your_mesh.points.flatten()
# axes.auto_scale_xyz(scale, scale, scale)
# filt = geom.chi
# axes.scatter(bcs.x[filt],bcs.y[filt],bcs.z[filt],'r')
# plt.show()

tht = 45*-2*np.pi/360
func1 = bcs.getRotFunc(-tht/2, ax = 1, long=0, offset = [-(NX+1)/2,0,-(NX+1)/2])
func2 = bcs.getRotFunc(tht/2, ax = 1, long=0, offset = [-(NX+1)/2,0,-(NX+1)/2])

bbox1 = [[0, NX+1], [0, 20], [0, NX + 1]]
bbox2 = [[0, NX+1], [geom.NY-20, geom.NY], [0, NX + 1]]
bcs.addFixedFunctional(bbox1, func1)
bcs.addFixedFunctional(bbox2, func2)

model.setBCs(bcs)
model.solve(10000, tol=None)

u,v,w = model.get_displacement()
xs = bcs.x
ys = bcs.y
zs = bcs.z
filt = geom.chi
mlab.points3d(xs[filt]+u[filt], ys[filt] + v[filt], zs[filt] + w[filt], scale_factor=.8)
mlab.show()
