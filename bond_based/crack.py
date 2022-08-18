import numpy as np
import modules.pd as pd
import sys
import time
import scipy.io
from matplotlib import pyplot as plt

dx = 0.0001
bbox = [[-.025, .025], [-.025 - 3*dx, .025 + 3*dx], [0,dx]]
E = 192e9
nu = 1/3
rho = 8000
NX = int((bbox[0][1]-bbox[0][0])/dx)
hrad = 3.015
ntau = 1000
vel = 50
ecrit = .04472

mat = pd.PDMaterial(E, nu, ecrit, rho)
geom = pd.PDGeometry(bbox, NX, None, hrad, 2)
model = pd.PDModel(mat, geom, None, dtype=np.float32, ntau=ntau, ADR=False, SCR=True, initcuts=True)
bcs = pd.PDBoundaryConditions(geom)
bbox1 = [[-.025, .025], [-.025-3*dx, -.025], [0, 0.0001]]
bbox2 = [[-.025, .025], [.025, .025+3*dx], [0, 0.0001]]
bcs.addFixed(bbox1, -vel*ntau*model.dt, [1])
bcs.addFixed(bbox2, vel*ntau*model.dt, [1])
model.setBCs(bcs)
model.solve(ntau)
xs = np.reshape(bcs.x, [NX,-1])
ys = np.reshape(bcs.y, [NX,-1])
dmg = np.reshape(model.d_Ft.get(), [NX,-1], order = 'F').T
c = plt.imshow(dmg)
plt.colorbar(c)
plt.show()
