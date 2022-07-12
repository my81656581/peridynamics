import numpy as np
import modules.display as disp
import modules.pd as pd

from matplotlib import pyplot as plt

# Continuum material properties
E = 2e11
nu = 0.3
rho = 1250.
mat = pd.PDMaterial(E, nu, rho)

# Geometry specifications
bbox = [[0, 1], [0, 1], [0, 1]]
NX = 32
hrad = 3.01
geom = pd.PDGeometry(bbox, NX, hrad)

# Boundary conditions
ntau = 100
appf = 1000000
bcs = pd.PDBoundaryConditions(geom, ntau)
# bcs.addDistributedForce([[0, .05], [0, 1], [0, 1]], appf*np.array([-1,0,0]))
# bcs.addDistributedForce([[.95, 1], [0, 1], [0, 1]], appf*np.array([1,0,0]))
bcs.addFixed([[0, .05], [0, 1], [0, 1]], 0, [0])
bcs.addFixed([[.95, 1], [0, 1], [0, 1]], 0.1, [0])

model = pd.PDModel(mat, geom, bcs, dtype=np.float32)
model.solve(1000)
# dply = disp.Display(geom, model)
# dply.launch_pyplot(num=500)

u, v, w = model.get_displacement()
x, y, z = model.get_coords()


xs = np.mean(np.mean(np.reshape(x,[geom.NZ,geom.NY,geom.NX]),axis=0),axis=0)
ums = np.mean(np.mean(np.reshape(u,[geom.NZ,geom.NY,geom.NX]),axis=0),axis=0)

ys = np.mean(np.mean(np.reshape(y,[geom.NZ,geom.NY,geom.NX]),axis=0),axis=1)
vms = np.mean(np.mean(np.reshape(v,[geom.NZ,geom.NY,geom.NX]),axis=0),axis=1)

zs = np.mean(np.mean(np.reshape(z,[geom.NZ,geom.NY,geom.NX]),axis=1),axis=1)
wms = np.mean(np.mean(np.reshape(w,[geom.NZ,geom.NY,geom.NX]),axis=1),axis=1)

ex, _ = np.linalg.lstsq(np.vstack([xs,np.ones(xs.shape)]).T, ums, rcond=None)[0]
ey, _ = np.linalg.lstsq(np.vstack([ys,np.ones(xs.shape)]).T, vms, rcond=None)[0]
ez, _ = np.linalg.lstsq(np.vstack([zs,np.ones(xs.shape)]).T, wms, rcond=None)[0]
print(ex, ey, ez, ey/ex)

plt.plot(xs, ums, '-o')
plt.plot(ys, vms, '-o')
plt.plot(zs, wms, '-o')
plt.show()