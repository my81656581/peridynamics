import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time
import scipy.io
from matplotlib import pyplot as plt
from mayavi import mlab

## Ankle HSA Experimental Comparison
# chi = scipy.io.loadmat('../geometries/ankle.mat')['in'].astype(np.bool_)
# whr = np.where(chi)
# chi = chi[np.min(whr[0]):np.max(whr[0])+1, np.min(whr[1]):np.max(whr[1])+1, np.min(whr[2]):np.max(whr[2])+1]
# scl = 100/300*1/1000
# bbox = [[0, scl*chi.shape[0]], [-scl*chi.shape[1]/2,scl*chi.shape[1]/2], [-scl*chi.shape[2]/2,scl*chi.shape[2]/2]]
# E = 7e8
# nu = 0.25
# NX = 314#int(bbox[0][1]-bbox[0][0])
# hrad = 1.5
# ntau = 200000 # 60000, 120000
# thtd = float(sys.argv[1])
# f = float(sys.argv[2])

# mat = pd.PDMaterial(E, nu)
# geom = pd.PDGeometry(bbox, NX, np.reshape(chi,[-1],order='F'), hrad)
# model = pd.PDModel(mat, geom, None, None, dtype=np.float32, ntau=ntau)
# bcs = pd.PDBoundaryConditions(geom)

# tht = thtd*-2*np.pi/360
# bcs = pd.PDBoundaryConditions(geom)
# func = bcs.getRotFunc(-tht/2, ax = 0)
# bbox1 = [[20*scl, 28*scl], [-30*scl, 30*scl], [-30*scl, 30*scl]]
# bbox2 = [[(314-28)*scl, (314-20)*scl], [-30*scl, 30*scl], [-30*scl, 30*scl]]
# bcs.addFixedFunctional(bbox1, func)
# func = bcs.getRotFunc(tht/2, ax = 0)
# bcs.addFixedFunctional(bbox2, func)

# bbox3 = [[31*scl, 34*scl], [-30*scl, 30*scl], [-30*scl, 30*scl]]
# bbox4 = [[(314-34)*scl, (314-31)*scl], [-30*scl, 30*scl], [-30*scl, 30*scl]]
# bcs.addDistributedForce(bbox3,[f,0,0])
# bcs.addDistributedForce(bbox4,[-f,0,0])
# model.setBCs(bcs)
# model.solve(300000, tol=None)

# u,v,w = model.get_displacement()
# xs = bcs.x
# ys = bcs.y
# zs = bcs.z
# filt = geom.chi
# etr = (np.max(xs[filt]+u[filt])-np.min(xs[filt]+u[filt]))/313/scl - 1
# F1 = model.eval_F(bbox1)
# F2 = model.eval_F(bbox2)
# print(thtd, etr, F2[0]- F1[0], f)

## Semi-Open HSA
chi = scipy.io.loadmat('../geometries/S-5D4L1_8_AN.mat')['in'].astype(np.bool_)
whr = np.where(chi)
chi = chi[np.min(whr[0]):np.max(whr[0])+1, np.min(whr[1]):np.max(whr[1])+1, np.min(whr[2]):np.max(whr[2])+1]
bbox = [[0, chi.shape[0]], [-chi.shape[1]/2,chi.shape[1]/2], [-chi.shape[2]/2,chi.shape[2]/2]]
E = 7e8
nu = 0.25
NX = int(bbox[0][1]-bbox[0][0])
hrad = 3.01
ntau = 1400
tht = -np.pi/2
mat = pd.PDMaterial(E, nu)
geom = pd.PDGeometry(bbox, NX, np.reshape(chi,[-1],order='F'), hrad)
bcs = pd.PDBoundaryConditions(geom)
func = bcs.getRotFunc(-tht/2, ax = 0)
bbox1 = [[0, 15], [-35, 35], [-35, 35]]
bbox2 = [[(166-15), 153], [-35, 35], [-35, 35]]
# bbox1 = [[0, 30], [-64, 64], [-64, 64]]
# bbox2 = [[(305-30), 305], [-64, 64], [-64, 64]]
bcs.addFixedFunctional(bbox1, func)
func = bcs.getRotFunc(tht/2, ax = 0)
bcs.addFixedFunctional(bbox2, func)
model = pd.PDModel(mat, geom, bcs, dtype=np.float32, ntau=ntau, mmlt=0.1)
model.solve(1700)
u,v,w = model.get_displacement()
xs = bcs.x
ys = bcs.y
zs = bcs.z
filt = geom.chi
print(np.sum(chi))
mlab.points3d(xs[filt]+u[filt], ys[filt] + v[filt], zs[filt] + w[filt], scale_factor=.8)
mlab.show()
