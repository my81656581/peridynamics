from numba import cuda,int64,float64
import numpy as np

TPB = 32

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
def d_calcPeriDot(dv, d_A, d_out):
	i = cuda.grid(1)

	if i < d_A.shape[1]:
		sm = 0
		for j in range(d_A.shape[0]):
			sm += d_A[j, i]
		d_out[i] = sm * dv

@cuda.jit
def d_calcElongation(dv, form, v, d_u, d_conn, d_eta, d_x, d_m, d_nu, d_e, d_dil):
	i = cuda.grid(1)

	if i<d_conn.shape[1]:
		ix, iy = d_u[:, i]
		sm = 0
		for j in range(d_conn.shape[0]):
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
def d_calcDevStrain(dv, d_e, d_dil, d_x, d_ed, d_wedx):
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
def d_calcForceState(E, p, q, v, form, dv, gfill, d_m, d_dil, d_x, d_ed, d_wedx, d_t, d_k, d_W, d_coord):
	i = cuda.grid(1)

	if i < d_x.shape[1]:
		ki = d_k[i]
		Ei = ki**p * E
		kp,am = 0., 0.
		k = Ei/(1-2*v)/3
		mu = Ei/(1+v)/2
		if form == 2:
			kp = k+mu/9
			am = 8*mu
		elif form == 1.:
			kp = k+mu/9*(v+1)**2/(2*v-1)**2
			am = 8*mu
		else:
			kp = k
			am = 15*mu

		mi = d_m[i]
		dil = d_dil[i]
		wedx = d_wedx[i]
		Wsm = kp * dil**2 / 2

		for j in range(d_x.shape[0]):
			xi = d_x[j, i]
			ed = d_ed[j, i]

			if xi > 0:
				w = 1/xi
			else:
				w = 0

			Wsm += am / mi / 2 * w * ed**2 * dv

			if form == 0:
				d_t[j,i] = 3*kp/mi + am/mi*w*ed
			elif form == 1:
				d_t[j,i] = 2*(2*v - 1)/(v - 1)*(kp*dil - am/mi/3 * wedx)/mi + am/mi * w * ed
			else:
				d_t[j,i] = 2*(kp*dil - am/mi/3*wedx)/mi + am/mi * w * ed

		xi, yi = d_coord[:,i]
		if (xi <0.05 and (yi<1/gfill or yi>(0.5 - 1/gfill))) or (xi>(1-1/gfill) and abs(yi-0.25)<(1/gfill)):
			d_W[i] = 0
		else:
			d_W[i] = Wsm**q

@cuda.jit
def d_calcForce(gfill, d_eta, d_nu, d_t, d_conn, d_iconn, d_F, d_Fp, d_Fa, d_coord):
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

		d_Fp[:, i] = d_F[0,i], d_F[1,i]
		d_F[:,i] = Fx, Fy
		xi, yi = d_coord[:,i]
		if (xi <0.05 and (yi<1/gfill or yi>(0.5 - 1/gfill))) or (xi>(1-1/gfill) and abs(yi-0.25)<(1/gfill)):
			d_Fa[i] = 0
		else:
			d_Fa[i] = (Fx*Fx)**0.5 + (Fy*Fy)**0.5

@cuda.jit
def d_fillCnd(d_u, d_F, d_up, d_Fp, d_cn, d_cd):
	i = cuda.grid(1)

	if i<d_u.shape[1]:
		nsm = 0
		dsm = 0
		for k in range(d_u.shape[0]):
			u = d_u[k,i]
			f = d_F[k,i]
			up = d_up[k,i]
			fp = d_Fp[k,i]
			nsm -= u**2*(f - fp)/(u - up)
			dsm += u**2
		d_cn[i] = nsm
		d_cd[i] = dsm

@cuda.jit
def d_stepU(gfill, dt, c, rho, d_F, d_u, d_up, d_coords, d_Fp):
	i = cuda.grid(1)

	if i<d_F.shape[1]:
		for k in range(d_F.shape[0]):
			# Apply essential boundary conditions
			if k == 0:
				xi, yi = d_coords[:,i]
				if (xi <0.05 and (yi<1/gfill or yi>(0.5 - 1/gfill))) or (xi>(1-1/gfill) and abs(yi-0.25)<(1/gfill)):
					return
			
			up = d_up[k, i]
			u = d_u[k,i]
			d_up[k,i] = u
			d_u[k,i] = (2*dt*dt*d_F[k,i]/rho + 4*u + (c*dt - 2)*up)/(2 + c*dt)

@cuda.reduce
def sum_reduce(a, b):
    return a + b

class PD:

	def __init__(self, geom, gfill, form, v, E, p, q, rho):
		NN, NB, ND = geom.d_coords.shape[1], geom.d_conn.shape[0], geom.d_coords.shape[0]

		# Initialize GPU arrays
		self.d_eta = cuda.to_device(np.zeros((ND, NB, NN)))
		self.d_x = cuda.to_device(np.zeros((NB, NN))) 
		self.d_m = cuda.to_device(np.zeros(NN))
		self.d_nu = cuda.to_device(np.zeros((ND, NB, NN)))
		self.d_e = cuda.to_device(np.zeros((NB, NN)))
		self.d_dil = cuda.to_device(np.zeros(NN))
		self.d_ed = cuda.to_device(np.zeros((NB, NN)))
		self.d_wedx = cuda.to_device(np.zeros(NN))
		self.d_t = cuda.to_device(np.zeros((NB, NN)))
		self.d_F = cuda.to_device(np.zeros((ND, NN)))
		self.d_Fp = cuda.to_device(np.zeros((ND, NN)))
		self.d_Fa = cuda.to_device(np.zeros(NN))
		self.d_cn = cuda.to_device(np.zeros(NN))
		self.d_cd = cuda.to_device(np.zeros(NN))
		self.d_W = cuda.to_device(np.zeros(NN))
		self.geom = geom
		self.fto = 1.
		self.gfill = gfill
		self.form = form
		self.v = v
		self.E = E
		self.p = p
		self.q = q
		self.rho = rho
		self.dt = 2 / 2 / gfill / gfill * (rho / E)**0.5
		self.NN = NN

		d_calcVector[((NN + TPB)//TPB, (NB + TPB)//TPB),(TPB, TPB)](geom.d_coords, geom.d_conn, self.d_eta)
		d_calcNorm[((NN + TPB)//TPB, (NB + TPB)//TPB),(TPB, TPB)](self.d_eta, self.d_x)
		d_calcPeriDot[(NN + TPB)//TPB, TPB](self.geom.dv, self.d_x, self.d_m)


	def solve(self, d_k, tol = 5e-3):
		ft = self.fto
		c = (self.gfill**2/self.dt)
		tt = 0
		while ft/self.fto>tol:
			d_calcElongation[(self.NN + TPB)//TPB,TPB](self.geom.dv, self.form, self.v, self.geom.d_u, self.geom.d_conn, self.d_eta, self.d_x, self.d_m, self.d_nu, self.d_e, self.d_dil)
			d_calcDevStrain[(self.NN + TPB)//TPB,TPB](self.geom.dv, self.d_e, self.d_dil, self.d_x, self.d_ed, self.d_wedx)
			d_calcForceState[(self.NN + TPB)//TPB,TPB](self.E, self.p, self.q, self.v, self.form, self.geom.dv, self.gfill, self.d_m, self.d_dil, self.d_x, self.d_ed, self.d_wedx, self.d_t, d_k, self.d_W, self.geom.d_coords)
			d_calcForce[(self.NN + TPB)//TPB,TPB](self.gfill, self.d_eta, self.d_nu, self.d_t, self.geom.d_conn, self.geom.d_iconn, self.d_F, self.d_Fp, self.d_Fa, self.geom.d_coords)
			if tt>0:
				d_fillCnd[(self.NN + TPB)//TPB,TPB](self.geom.d_u, self.d_F, self.geom.d_up, self.d_Fp, self.d_cn, self.d_cd)
				c = 2*(max(0,sum_reduce(self.d_cn))/sum_reduce(self.d_cd))**0.5
			d_stepU[(self.NN + TPB)//TPB,TPB](self.gfill, self.dt, c, self.rho, self.d_F, self.geom.d_u, self.geom.d_up, self.geom.d_coords, self.d_Fp)
			ft = sum_reduce(self.d_Fa)
			if tt==0 and self.fto == 1:
				self.fto = ft
			tt += 1