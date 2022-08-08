import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time
from mayavi import mlab
from matplotlib import pyplot as plt

# 3D Cantilever, bottom edge loaded
# bbox = [[0, 2], [0, 1], [0, 1]]
# NX = 80 # 80, 90, 100
# E = 1
# nu = 0.25
# hrad = 3.01
# rho = 0.1
# ntau = 500
# volfrac = 0.25
# penal = 4
# alpha = 0.7
# mat = pd.PDMaterial(E, nu, rho)
# geom = pd.PDGeometry(bbox, NX, None, hrad, dim=3)
# bcs = pd.PDBoundaryConditions(geom)
# bcs.addFixed([[0, .1], [0, 1], [0, 1]], 0, [0,1,2])
# bcs.addDistributedForce([[1.95,2], [0, .05], [0, 1]], [0,-0.0005,0])

# 3D Tiered Bridge
# bbox = [[0, 90], [0, 45], [0, 20]]
# NX = 180
# E = 210e9
# nu = 0.25
# hrad = 3.01
# rho = 1250
# ntau = 1000
# volfrac = 0.17
# penal = 3
# alpha = 0.5
# mat = pd.PDMaterial(E, nu, rho)
# geom = pd.PDGeometry(bbox, NX, None, hrad, dim=3)
# bcs = pd.PDBoundaryConditions(geom)
# x = bcs.x
# y = bcs.y
# z = bcs.z
# geom.chi = ~((y>22) & (y<34) & (z>5) & (z<15))
# bcs.addFixed([[0, 1], [0, 1], [0, 20]], 0, [0,1,2])
# bcs.addFixed([[89,90], [0, 45], [0, 20]], 0, [0])
# bcs.addDistributedForce([[0, 90], [20, 22], [0, 20]], [0,-10000000000,0])

# 3D Gripper Mechanism
# bbox = [[0, 80], [0, 40], [0, 20]]
# NX = 80
# E = 3e9
# nu = 0.25
# hrad = 3.01
# rho = 1250
# ntau = 1000
# volfrac = 0.17
# penal = 3
# alpha = 0.3
# mat = pd.PDMaterial(E, nu, rho)
# geom = pd.PDGeometry(bbox, NX, None, hrad, dim=3)
# bcs = pd.PDBoundaryConditions(geom)
# x = bcs.x
# y = bcs.y
# z = bcs.z
# geom.chi = ~((y<8) & (y>-8) & (x>64))
# bcs.addFixed([[0, 80], [0, 1], [0, 20]], 0, [1])
# bcs.addFixed([[0, 1], [39, 40], [0, 20]], 0, [0,1,2])
# bcs.addDistributedForce([[0, 1], [1, 2], [0, 20]], [10000000,0,0])
# bcs.addDistributedForce([[79, 80], [9, 10], [0, 20]], [0,10000000,0])

# Electric Tower
# bbox = [[0, 10], [0, 25], [0, 5]]
# NX = 60
# E = 3e9
# nu = 0.25
# hrad = 3.01
# rho = 100
# ntau = 1000
# volfrac = 0.05
# penal = 3
# alpha = 0.6
# mat = pd.PDMaterial(E, nu, rho)
# geom = pd.PDGeometry(bbox, NX, None, hrad, dim=3)
# bcs = pd.PDBoundaryConditions(geom)
# x = bcs.x
# y = bcs.y
# z = bcs.z
# geom.chi = ~((y<20) & ((x<5) | (x>15)))
# bcs.addFixed([[5, 5.5], [0, 0.5], [0, 0.5]], 0, [0,1,2])
# bcs.addFixed([[5, 5.5], [0, 0.5], [4.5, 5]], 0, [0,1,2])
# bcs.addDistributedForce([[0, 0.5], [20, 20.5], [2, 3]], [0,-3000000,0])
# bcs.addFixed([[9.5, 10], [0, 25], [0, 5]], 0, [0])


# opt = pd.PDOptimizer(alpha, volfrac, penal)
# model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float32,ntau=ntau)
# # dply = disp.Display(geom, model, opt, tol = 0.01, num=50000)
# # dply.launch()
# for _ in range(50):
#     model.solve(50000, tol = 0.01, minT = 100)
#     opt.step(model)
# vals = model.get_fill()
# np.save("..\\auxiliary\\cantilever.npy",vals)
# vals[vals<0.95] = 0
# vals = np.reshape(vals,(geom.NZ,geom.NY,geom.NX)).T/np.max(vals)
# fig = mlab.figure(size=(600,600))
# vox = mlab.pipeline.volume(mlab.pipeline.scalar_field(vals), vmin=0, vmax=1)
# mlab.show()


bbox = [[0, 160], [0, 20], [0, 3]]
NX = 2000
E = 30e6
nu = 0.25
rho = 1250.
hrad = 3.01
ntau = 1000
volfrac = 0.2
penal = 2
alpha = 0.6
mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, None, hrad, dim=2)
bcs = pd.PDBoundaryConditions(geom)
bcs.addFixed([[0, 1], [0, 20], [0, 10]], 0, [0,1,2])
bcs.addFixed([[159, 160], [0, 20], [0, 10]], 0, [0,1,2])
bcs.addFixed([[79,81], [0, 1], [0, 10]], -.4, [1])
opt = pd.PDOptimizer(alpha, volfrac, penal)
model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float32,ntau=ntau)

t0 = time.time()
for _ in range(50):
    model.solve(50000, tol = 0.01)
    opt.step(model)
tf = time.time()-t0
print("Avg time of ", geom.NN, ": ",tf/50)
# dply = disp.Display(geom, model, opt, tol = 0.01)
# dply.launch_2D()

NX, NY = geom.NX, geom.NY
fig, ax = plt.subplots(1)
heatmap = ax.pcolor(np.reshape(model.get_fill(), (NY, NX)))
ax.set_aspect('equal', 'box')
fig.tight_layout()
plt.show()
