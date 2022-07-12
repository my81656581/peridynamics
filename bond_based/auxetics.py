import numpy as np
import modules.display as disp
import modules.pd as pd
import sys
import time
import scipy.io
from matplotlib import pyplot as plt


chi = scipy.io.loadmat('sphere1_128.mat')['in'].astype(np.bool_)
whr = np.where(chi)
chi = chi[np.min(whr[0]):np.max(whr[0])+1, np.min(whr[1]):np.max(whr[1])+1, np.min(whr[2]):np.max(whr[2])+1]
bbox = [[-chi.shape[0]/2,chi.shape[0]/2], [-chi.shape[1]/2,chi.shape[1]/2],[0, chi.shape[2]]]

E = 3e9
nu = 0.25
NX = int(bbox[0][1]-bbox[0][0])
hrad = 3.01
rho = 1250.
ntau = 10000
tht = -np.pi/4

mat = pd.PDMaterial(E, nu, rho)
geom = pd.PDGeometry(bbox, NX, np.reshape(chi,[-1],order='F'), hrad)
bcs = pd.PDBoundaryConditions(geom, ntau)
bcs.addFixed([[-62, 62], [-62, 62], [0, 3]], 0, [0,1,2])
func = bcs.getRotFunc(tht)
bcs.addFixedFunctional([[-62, 62], [-62, 62], [115, 125]], func)

model = pd.PDModel(mat, geom, bcs, dtype=np.float32)
# dply = disp.Display(geom, model)
# dply.launch_pyplot()

model.solve(20000, tol=0.001)
u,v,w = model.get_displacement()
xs = bcs.x
ys = bcs.y
zs = bcs.z
filt = geom.chi
M = 1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs[filt]+M*u[filt], ys[filt] + M*v[filt], zs[filt] + M*w[filt],'o', markersize=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()