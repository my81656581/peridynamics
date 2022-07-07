import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time

# NX = int(sys.argv[1])

# bbox = [[0, 80], [0, 40], [0, 5]]
# E = 2e11
# nu = 0.333
# volfrac = 0.5
# penal = 3
# NX = int(bbox[0][1])

bbox = [[0, 160], [0, 20], [0, 3]]
E = 30e6
nu = 0.3
volfrac = 0.2
penal = 3
NX = 2*int(bbox[0][1])

hrad = 3.01
rho = 1250.
ntau = 500
alpha = 0.3

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, hrad)
bcs = pd.PDBoundaryConditions(geom, ntau)
# bcs.addFixed([[0, 1], [0, 40], [0, 5]], 0, [0,1,2])
# bcs.addFixed([[79,80], [19, 21], [0, 5]], -0.1, [1])

bcs.addFixed([[0, 1], [0, 20], [0, 10]], 0, [0,1,2])
bcs.addFixed([[159, 160], [0, 20], [0, 10]], 0, [0,1,2])
bcs.addFixed([[79,81], [0, 1], [0, 10]], -0.1, [1])
# bcs.addDistributedForce([[79,81], [0, 1], [0, 5]], 1000000*np.array([0,-1,0]))

opt = pd.PDOptimizer(alpha, volfrac, penal)
model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float64)

# t0 = time.time()
# model.solve(5000)
# model.get_fill()
# print(geom.NN, time.time()-t0)

dply = disp.Display(geom, model, opt)
dply.launch()
while True:
    dply.launch_pyplot()
    opt.step(model)