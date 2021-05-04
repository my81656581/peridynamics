import os
os.environ["NUMBA_ENABLE_CUDASIM"] = '0'

from numba import cuda,int64,float64
import numpy as np
import math
# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
import sys
import time

# Formulation:  0: 3D; 1: Plane stress; 2: Plane strain
form = 2
tb = 1. #thickness
ln = 1. #length
wd = 0.5 #width

# Grid density (~nodes per unit length)
gfill = int(sys.argv[1])

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
dt = 1e-10

TPB = 32

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

@cuda.jit
def d_getConn(d_coords, d_conn):
	i = cuda.grid(1)

	if i<d_coords.shape[1]:
		dsti = cuda.local.array(NH,float64)
		coni = cuda.local.array(NH, int64)
		for j in range(d_conn.shape[0]):
			coni[j] = i
		
		wdst, wind, aind = -1.,-1, 0
		xi, yi = d_coords[:, i]
		for j in range(d_coords.shape[1]):
			xj, yj = d_coords[:, j]
			dst = ((xi - xj)**2 + (yi - yj)**2)**0.5
			
			if dst<hrad:
				if aind<d_conn.shape[0]:
					coni[aind] = j
					dsti[aind] = dst
					if dst>wdst:
						wdst = dst
						wind = aind
					aind += 1
				elif dst<wdst:
					coni[wind] = j
					dsti[wind] = dst
					wdst = dst
					for k in range(d_conn.shape[0]):
						dstk = dsti[k]
						if dstk>wdst:
							wdst = dstk
							wind = k

		for j in range(d_conn.shape[0]):
			d_conn[j, i] = coni[j]

@cuda.jit
def d_getInvConn(d_conn, d_iconn):
	# conn[j,i]: index of jth bond of i
	# iconn[j,i]: conn[iconn[j,i], conn[j,i]] = i
	i, j = cuda.grid(2)

	if j<d_conn.shape[0] and i<d_conn.shape[1]:
		jind = d_conn[j, i]
		
		if jind == i:
			d_iconn[j, i] = 0
		else:
			kind = 0
			for k in range(d_conn.shape[0]):
				if d_conn[k, jind] == i:
					kind = k
			d_iconn[j, i] = kind

# Generate geometry
coords = getCartCoords()
ND, NB, NN = coords.shape[0],NH,coords.shape[1]
d_coords = cuda.to_device(coords)
d_conn = cuda.to_device(np.zeros((NB, coords.shape[1])).astype(np.int32))
d_getConn[(NN + TPB)//TPB, TPB](d_coords,d_conn)
d_iconn = cuda.to_device(np.zeros((NB, NN)).astype(int) - 1)
d_getInvConn[((NN + TPB)//TPB, (NB + TPB)//TPB),(TPB, TPB)](d_conn, d_iconn)
dv = tb*ln*wd / coords.shape[1]

conn = d_conn.copy_to_host()
iconn = d_iconn.copy_to_host()
# np.save('/usr/lusers/jdbart/coords.npy',coords)
# np.save('/usr/lusers/jdbart/conn.npy',conn)
# np.save('/usr/lusers/jdbart/iconn.npy',iconn)

@cuda.jit
def d_calcNorm(d_vectors, d_out):
	i = cuda.grid(1)

	if i<d_vectors.shape[2]:
		for j in range(d_vectors.shape[1]):
			sm = 0
			for k in range(d_vectors.shape[0]):
				sm += d_vectors[k,j,i]**2
			d_out[j,i] = sm**0.5

@cuda.jit
def d_calcVector(d_coords, d_conn, d_out):
	i = cuda.grid(1)

	if i<d_coords.shape[1]:
		for j in range(d_conn.shape[0]):
			jind = d_conn[j, i]
			for k in range(d_coords.shape[0]):
				d_out[k, j, i] = d_coords[k, jind] - d_coords[k, i]

@cuda.jit
def d_calcPeriDot(d_A, d_out):
	i = cuda.grid(1)

	if i < d_A.shape[1]:
		sm = 0
		for j in range(d_A.shape[0]):
			sm += d_A[j, i]
		d_out[i] = sm * dv

@cuda.jit
def d_calcElongation(d_u, d_conn, d_eta, d_x, d_m, d_nu, d_e, d_dil):
	i = cuda.grid(1)

	if i<NN:
		ix, iy = d_u[:, i]
		sm = 0
		for j in range(NB):
			jind = d_conn[j, i]
			
			nu_x = d_u[0, jind] - ix
			nu_y = d_u[1, jind] - iy

			d_nu[0,j,i] = nu_x
			d_nu[1,j,i] = nu_y
			xji = d_x[j,i]
			e_ij = ((d_eta[0,j,i] + nu_x)**2 + (d_eta[1,j,i] + nu_y)**2)**0.5 - xji
			d_e[j, i] = e_ij
			sm += e_ij
		
		PD_e = sm * dv
		if form == 0:
			dil = 3 / d_m[i] * PD_e
		elif form == 1:
			dil = 2*(2*v - 1)/(v - 1) * PD_e / d_m[i]
		else:
			dil = 2 / d_m[i] * PD_e

		d_dil[i] = dil

@cuda.jit
def d_calcDevStrain(d_e, d_dil, d_x, d_ed, d_wedx):
	i = cuda.grid(1)

	if i < d_e.shape[1]:
		sm = 0
		dil = d_dil[i]
		for j in range(d_e.shape[0]):
			edj = d_e[j, i] - dil * d_x[j, i] / 3
			d_ed[j, i] = edj
			sm += edj
		d_wedx[i] = sm*dv


@cuda.jit
def d_calcForceState(d_m, d_dil, d_x, d_ed, d_wedx, d_t):
	i = cuda.grid(1)

	if i < d_x.shape[1]:
		for j in range(d_x.shape[0]):
			mi = d_m[i]
			xi = d_x[j, i]
			if xi > 0:
				w = 1/xi
			else:
				w = 0

			if form == 0:
				d_t[j,i] = 3*kp/mi + am/mi*w*d_ed[j, i]
			elif form == 1:
				d_t[j,i] = 2*(2*v - 1)/(v - 1)*(kp*d_dil[i] - am/mi/3 * d_wedx[i])/mi + am/mi * w * d_ed[j, i]
			else:
				d_t[j,i] = 2*(kp*d_dil[i] - am/mi/3*d_wedx[i])/mi + am/mi * w * d_ed[j, i]


@cuda.jit
def d_calcForce(d_eta, d_nu, d_t, d_conn, d_iconn, d_F):
	i = cuda.grid(1)

	if i < d_eta.shape[2]:
		Fx, Fy = 0, 0
		for j in range(d_eta.shape[1]):
			fij = d_t[j, i] + d_t[d_iconn[j,i], d_conn[j,i]]

			ex, ey = d_eta[:,j,i]
			nx, ny = d_nu[:,j,i]

			enn = ((ex + nx)**2 + (ey + ny)**2)**0.5
			if enn>0:
				Fx += fij*(ex + nx)/enn
				Fy += fij*(ey + ny)/enn

		d_F[:,i] = Fx, Fy

@cuda.jit
def d_stepU(d_F, d_u, d_v, d_coord):
	i = cuda.grid(1)

	if i<d_F.shape[1]:
		for k in range(d_F.shape[0]):
			if k == 0:
				xi = d_coord[0,i]
				if xi <0.05 or xi>0.95:
					return
			vi = d_v[k, i]
			vn = vi + dt/rho*(d_F[k,i] + fdc*rho*vi)
			d_v[k,i] = vn
			d_u[k,i] += dt*vn

# Apply initial dispacements
u = np.zeros(coords.shape)
u[0,coords[0,:]>0.95] = 0.1
vel = np.zeros(coords.shape)
d_u = cuda.to_device(u)
d_v = cuda.to_device(vel)

# Initialize GPU arrays
d_eta = cuda.to_device(np.zeros((ND, NB, NN)))
d_x = cuda.to_device(np.zeros((NB, NN))) 
d_m = cuda.to_device(np.zeros(NN))

d_nu = cuda.to_device(np.zeros((ND, NB, NN)))
d_e = cuda.to_device(np.zeros((NB, NN)))
d_PD_e = cuda.to_device(np.zeros(NN))
d_dil = cuda.to_device(np.zeros(NN))
d_ed = cuda.to_device(np.zeros((NB, NN)))
d_wedx = cuda.to_device(np.zeros(NN))
d_t = cuda.to_device(np.zeros((NB, NN)))
d_F = cuda.to_device(np.zeros((ND, NN)))

# Simplification: w = 1/x; wx = 1
d_calcVector[((NN + TPB)//TPB, (NB + TPB)//TPB),(TPB, TPB)](d_coords, d_conn, d_eta)
d_calcNorm[((NN + TPB)//TPB, (NB + TPB)//TPB),(TPB, TPB)](d_eta, d_x)
d_calcPeriDot[(NN + TPB)//TPB, TPB](d_x, d_m)

t0 = time.time()
for tt in range(1000):
	d_calcElongation[(NN + TPB)//TPB,TPB](d_u, d_conn, d_eta, d_x, d_m, d_nu, d_e, d_dil)
	d_calcDevStrain[(NN + TPB)//TPB,TPB](d_e, d_dil, d_x, d_ed, d_wedx)
	d_calcForceState[(NN + TPB)//TPB,TPB](d_m, d_dil, d_x, d_ed, d_wedx, d_t)
	d_calcForce[(NN + TPB)//TPB,TPB](d_eta, d_nu, d_t, d_conn, d_iconn, d_F)
	d_stepU[(NN + TPB)//TPB,TPB](d_F, d_u, d_v, d_coords)

print(NN, (time.time() -t0)/1000)

# u = d_u.copy_to_host()
# np.save('/usr/lusers/jdbart/u.npy',u)
# plt.plot(coords[0,:], coords[1,:], 'bo')
# plt.plot(coords[0,:] + u[0,:], coords[1,:] + u[1,:], 'kx')
# plt.show()