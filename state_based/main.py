import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time

# NX = int(sys.argv[1])

# bbox = [[0, 80], [0, 20], [0, 5]]
# E = 3e9
# nu = 0.25
# volfrac = 0.5
# penal = 3
# NX = int(8*bbox[0][1])

bbox = [[0, 160], [0, 20], [0, 3]]
E = 30e6
nu = 0.25
volfrac = 0.2
penal = 2
NX = 2*int(bbox[0][1])

hrad = 3.01
rho = 1250.
ntau = 1000
alpha = 0.8

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, None, hrad)
bcs = pd.PDBoundaryConditions(geom, ntau)
# bcs.addFixed([[0, 1], [0, 40], [0, 5]], 0, [0,1,2])
# bcs.addFixed([[79,80], [9, 11], [0, 5]], -1, [1])

bcs.addFixed([[0, 1], [0, 20], [0, 10]], 0, [0,1,2])
bcs.addFixed([[159, 160], [0, 20], [0, 10]], 0, [0,1,2])
bcs.addFixed([[79,81], [0, 1], [0, 10]], -.4, [1])

opt = pd.PDOptimizer(alpha, volfrac, penal)
model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float32)

# model.solve(1, tol = 0.006)

dply = disp.Display(geom, model, opt, tol = 0.01)
dply.launch()
