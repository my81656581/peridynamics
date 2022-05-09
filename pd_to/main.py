import numpy as np
import modules.disp as disp
import modules.pd as pd
import sys
import time

NX = int(sys.argv[1])
bbox = [[0, 2], [0, 1], [0, 1]] # Bounding box for geometry
hrad = 3.01
E = 2e11
nu = 0.3
rho = 1250.
ntau = 500
alpha = 0.3
volfrac = 0.4

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, hrad)
bcs = pd.PDBoundaryConditions(geom, ntau)
bcs.addFixed([[0, .05], [0, 1], [0, 1]], 0, [0,1,2])
# bcs.addFixed([[1.95, 2], [0, 1], [0, 1]], 0, [0,1,2])
bcs.addFixed([[1.95, 2.0], [.45, .55], [0, 1]], -0.1, [1])
opt = pd.PDOptimizer(alpha, volfrac)
model = pd.PDModel(mat, geom, bcs, opt, dtype=np.float64)

# t0 = time.time()
# model.solve(1000)
# model.get_fill()
# print(geom.NN, time.time()-t0)

dply = disp.Display(geom, model, 500)
dply.launch()