import modules.disp as disp
import modules.geom as geom
import modules.pd as pd
import modules.to as to

import sys
import time

# Geometry Parameters
tb = 1. #thickness
ln = 1. #length
wd = 0.5 #width
gfill = int(sys.argv[1]) # Grid density (~nodes per unit length)
NB = 37 # Number of bonds per node

# Peridynamic Parameters
form = 2 # Formulation:  0: 3D; 1: Plane stress; 2: Plane strain
E = 91.e9 #Young's Modulus
v = 1/3 #Poisson's Ratio
rho = 1. #Density

# Optimization parameters
p = 2
q = 1
alpha = 0.4
volfrac = 0.5

# Animate?
anm = True



# Build geometry
geometry = geom.Geom(tb, ln, wd, gfill, NB)
# Initialize PD model
pdmod = pd.PD(geometry, gfill, form, v, E, p, q, rho)
# Initialize Optimizer
opt = to.TO(geometry, pdmod, volfrac, alpha)
# Set up animation
if anm:
	plot = disp.Disp(geometry)



print("Beginning simulation")
for nl in range(100):
	pdmod.solve(opt.d_k)
	opt.step()
	if anm:
		plot.update(opt.d_k)

