import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time

NX = int(sys.argv[1])
# bbox = [[-.5, .5], [-.5, .5], [0, 1]] # Bounding box for geometry
bbox = [[0, 100], [0, 50], [0, 7]]
hrad = 3.01
E = 2e11
nu = 0.25
rho = 1250.
ntau = 500
alpha = 0.3
volfrac = 0.5
penal = 2

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, hrad)
bcs = pd.PDBoundaryConditions(geom, ntau)
bcs.addFixed([[0, 1], [0, 50], [0, 7]], 0, [0,1,2])
bcs.addFixed([[99,100], [24, 26], [0, 7]], -0.001, [1])
# bcs.addDistributedForce([[99,100], [24, 26], [0, 7]], 10000000*np.array([0,-1,0]))

# bcs.addFixed([[-.5, .5], [-.5, .5], [0, 0.05]], 0, [0,1,2])
# tht = np.pi/20
# def u_tor(x, y, z):
#     return x*np.cos(tht) - y*np.sin(tht) - x,   \
#             x*np.sin(tht) + y*np.cos(tht) - y,   \
#             np.zeros(z.shape)

# bcs.addFixedFunctional([[-.5, .5], [-.5, .5], [.95, 1]], u_tor)
        


opt = pd.PDOptimizer(alpha, volfrac, penal)
model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float64)

# t0 = time.time()
# model.solve(2000)
# model.get_fill()
# print(geom.NN, time.time()-t0)

dply = disp.Display(geom, model)
dply.launch()
# dply.launch_pyplot()