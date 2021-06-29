from numba import cuda,int64,float64
import numpy as np

TPB = 32

@cuda.jit
def d_zeroKbar(d_kbar):
	i = cuda.grid(1)

	if i<d_kbar.shape[0]:
		d_kbar[i] = 0

@cuda.jit
def d_calcKopt(RM, Wt, d_W, d_kopt):
	i = cuda.grid(1)

	NN = d_W.shape[0]
	if i<NN:
		d_kopt[i] = NN * RM * d_W[i] / Wt

@cuda.jit
def d_calcKbar(hrad, d_kopt, d_conn, d_x, d_kbar):
	i = cuda.grid(1)

	if i<d_conn.shape[1]:
		nsm, dsm = 0., 0.
		for j in range(d_conn.shape[0]):
			xij = d_x[j, i]
			if xij > .1*hrad: # (Make sure this isn't fake connection)
				psi = max(0, hrad - xij)
				nsm += psi * d_kopt[d_conn[j, i]]
				dsm += psi

		d_kbar[i] = max(0.000001, min(1, d_kbar[i] + nsm / dsm))

@cuda.jit
def d_updateK(gfill, alpha, d_kbar, d_k, d_coord):
	i = cuda.grid(1)

	if i<d_k.shape[0]:
		xi, yi = d_coord[:,i]
		if (xi <0.05 and (yi<1/gfill or yi>(0.5 - 1/gfill))) or (xi>(1-1/gfill) and abs(yi-0.25)<(1/gfill)):
			return
		d_k[i] = alpha*d_k[i] + (1 - alpha) * d_kbar[i]

@cuda.reduce
def sum_reduce(a, b):
    return a + b

class TO:
	def __init__(self, geom, pd, volfrac, alpha):
		NN = geom.d_coords.shape[1]
		self.d_k = cuda.to_device(np.ones(NN))
		self.d_kopt = cuda.to_device(np.ones(NN))
		self.d_kbar = cuda.to_device(np.ones(NN))
		self.pd = pd
		self.geom = geom
		self.volfrac = volfrac
		self.hrad = 3.52/self.pd.gfill
		self.alpha = alpha
		self.NN = NN

	def step(self, tol = 0.01):
		RM = self.volfrac
		d_zeroKbar[(self.NN + TPB)//TPB,TPB](self.d_kbar)
		while RM > tol:
			Wt = sum_reduce(self.pd.d_W)
			d_calcKopt[(self.NN + TPB)//TPB,TPB](RM, Wt, self.pd.d_W, self.d_kopt)
			d_calcKbar[(self.NN + TPB)//TPB,TPB](self.hrad, self.d_kopt, self.geom.d_conn, self.pd.d_x, self.d_kbar)
			AM = sum_reduce(self.d_kbar)/self.NN
			RM = self.volfrac - AM
		d_updateK[(self.NN + TPB)//TPB,TPB](self.pd.gfill, self.alpha, self.d_kbar, self.d_k, self.geom.d_coords)