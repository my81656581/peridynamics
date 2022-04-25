import modules.pd as pd
import numpy as np
import modules.disp as disp
import sys

NX = int(sys.argv[1])
bbox = [[0, 1], [0, 1], [0, 1]] # Bounding box for geometry
E = 2e11
nu = 0.3
rho = 1250.
ntau = 500

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX)
bcs = pd.PDBoundaryConditions(geom, ntau)
bcs.addFixed([[0, .05], [0, 1], [0, 1]], 0, [0])
bcs.addFixed([[.95, 1], [0, 1], [0, 1]], 0.1, [0])
model = pd.PDModel(mat, geom, bcs, dtype=np.float32)
dply = disp.Display(geom, model)
# dply.launch()