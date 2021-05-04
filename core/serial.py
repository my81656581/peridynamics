import os
os.environ["NUMBA_ENABLE_CUDASIM"] = '0'

from numba import cuda,int64,float64
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# Formulation:  0: 3D; 1: Plane stress; 2: Plane strain
form = 2
tb = 1. #thickness
ln = 1. #length
wd = 0.5 #width

# Grid density (~nodes per unit length)
gfill = 10

hrad = 3.52/gfill
NH = 37

#Material properties
E = 91.e9 #Young's Modulus
v = 1/3 #Poisson's Ratio
nu = v
k = E/(1-2*v)/3 #Bulk Modulus
mu = E/(1+v)/2 #Shear Modulus
rho = 1. #Density

#Derived peridynamic material properties
kp,am = 0., 0. #am = alpha*mi in Le & Bobaru
if form == 2:
	kp = k+mu/9
	am = 8*mu
elif form == 1.:
	kp = k+mu/9*(v+1)**2/(2*v-1)**2
	am = 8*mu
else:
	kp = k
	am = 15*mu

# dr = 1e-200
# h = 1/gfill
# fdc = -2*dr*np.pi/h*np.sqrt(E/3/rho/(1-2*v)) # Fdamp = fdc*nmass*vi
# dt = h/np.pi*(np.sqrt(1+dr**2)-dr)*np.sqrt(rho*3*(1-2*v)/E)/2
# print("DT: ", dt)
fdc = 0.1
dt = 1e-9

def calcChi(xs, ys):
	chi = np.zeros(xs.shape)
	chi[(xs>=-hrad) & (ys>=-hrad) & (xs<=1+hrad) & (ys<=0.5+hrad)] = 2
	chi[(xs>=0) & (ys>=0) & (xs<=1) & (ys<=0.5)] = 1
	return chi

def getCartCoords():
	L = 1/gfill
	grid_bbox = [[-.2 - L/2, 1.2],[-.2 - L/2, .7]]
	xs = np.arange(grid_bbox[0][0],grid_bbox[0][1], L)
	ys = np.arange(grid_bbox[1][0],grid_bbox[1][1], L)
	XS, YS = np.meshgrid(xs, ys)
	xs, ys = np.reshape(XS,[1, -1]), np.reshape(YS, [1, -1])
	chi = calcChi(xs, ys)
	rcoords = np.vstack([xs[chi == 1], ys[chi == 1]])
	#fcoords = np.vstack([xs[chi == 2], ys[chi == 2]])
	return rcoords#, fcoords

def getConn(rcoords):
	NR = rcoords.shape[1]

	rconn = np.zeros((NH, NR)).astype(int)

	txs = rcoords[0:1,:]
	tys = rcoords[1:,:]

	dst = ((txs - txs.T)**2 + (tys - tys.T)**2)**0.5

	for i in range(NR):
		coni = np.argsort(dst[i,:])[:NH]
		dsti = dst[i,coni]

		coni[dsti>hrad] = i
		rconn[:,i] = coni

	return rconn

def getInvConn(conn):
	# conn[j,i]: index of jth bond of i
	# iconn[j,i]: conn[iconn[j,i], conn[j,i]] = i
	iconn = np.zeros(conn.shape).astype(int) - 1

	for i in range(conn.shape[1]):
		for j in range(conn.shape[0]):
			jind = conn[j, i]
			if jind != i:
				iconn[j,i] = np.argwhere(conn[:,jind] == i)
			else:
				iconn[j,i] = 0
	return iconn

# Generate geometry
coords = getCartCoords()
conn = getConn(coords)
iconn = getInvConn(conn)
dv = tb*ln*wd / coords.shape[1]


def calcNorm(vectors):
	return np.sqrt(np.sum(vectors**2, axis=0)) # Shape: conn, ind

def calcVector(coords, conn):
	eta = coords[:, conn] - np.expand_dims(coords,1)
	return eta # Shape: x/y, conn, ind

def calcInfluence(x):
	w = 1/x
	w[np.isinf(w)] = 0
	return w

def calcElongation(eta, nu, x):
	return calcNorm(eta + nu) - x

def calcPeriDot(Ac, Bc, dV = dv):
	# either Ac or Bc should have same shape as conn (ie, be a PD state)
	return np.sum(Ac*Bc*dV, axis = 0)

def calcDilation(PD_e, m):
	if form == 0:
		return 3 / m * PD_e
	elif form == 1:
		return 2*(2*v - 1)/(v - 1) * PD_e / m
	else:
		return 2 / m * PD_e

def calcDevStrain(e, dil, x):
	return e - dil * x / 3

def calcForceState(m, dil, w, ed, wedx):
	if form == 0:
		return 3*kp/m + am/m*w*ed
	elif form == 1:
		return 2*(2*v - 1)/(v - 1)*(kp*dil - am/m/3 * wedx)/m + am/m * w*ed
	else:
		return 2*(kp*dil - am/m/3*wedx)/m + am/m * w * ed

def calcForce(eta, nu, t, conn, iconn):
	f = t + t[iconn, conn]
	en = eta + nu
	enn = calcNorm(en)
	enn[enn==0] = 1
	F = np.sum(f/enn * en, axis = 1)
	return F

def stepU(F, u, v):
	v += dt/rho*(F + fdc*rho*v)
	u += dt*v

	u[0,coords[0,:]<0.05] = 0.
	u[0,coords[0,:]>0.95] = 0.1
	v[0,coords[0,:]<0.05] = 0.
	v[0,coords[0,:]>0.95] = 0.

# Apply initial dispacements
u = np.zeros(coords.shape)
u[0,coords[0,:]>0.95] = 0.1
vel = np.zeros(coords.shape)

# Simplification: w = 1/x; wx = 1
eta = calcVector(coords, conn)
x = calcNorm(eta)
w = calcInfluence(x)
m = calcPeriDot(1, x)

for tt in range(1000):
	nu = calcVector(u, conn)
	e = calcElongation(eta, nu, x)
	PD_e = calcPeriDot(1, e)
	dil = calcDilation(PD_e, m)
	ed = calcDevStrain(e, dil, x)
	wedx = calcPeriDot(ed, 1)
	t = calcForceState(m, dil, w, ed, wedx)
	F = calcForce(eta, nu, t, conn, iconn)
	stepU(F, u, v)