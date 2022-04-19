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
appu = 0.1

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX)
#TODO: Generalize boundary conditions
bcs = pd.PDBoundaryConditions(appu, ntau)
model = pd.PDModel(mat, geom, bcs, dtype=np.float64)
dply = disp.Display(geom, model)
dply.launch()