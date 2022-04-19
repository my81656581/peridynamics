import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time


def makeStencil(hrad = 3.01):
	grad = int(hrad+1)
	xs = np.arange(-grad, grad+1, dtype=np.int32)
	XS,YS,ZS = np.meshgrid(xs,xs,xs)
	xs = np.reshape(XS, [-1])
	ys = np.reshape(YS, [-1])
	zs = np.reshape(ZS, [-1])
	rad = (xs**2 + ys**2 + zs**2)**0.5 
	filt = (rad < hrad) & (rad>0.001)
	S = np.vstack([xs[filt], ys[filt], zs[filt]]).T
	return S

class PDMaterial:
	def __init__(self, E, nu, rho, tsi = 1, ecrit = 100000):
		self.E = E
		self.nu = nu
		self.rho = rho
		self.tsi = tsi
		self.ecrit = ecrit

class PDGeometry:
	def __init__(self, bbox, NX, hrad=3.01):
		L = np.float64((bbox[0][1] - bbox[0][0])/NX)
		NY = int(np.round((bbox[1][1] - bbox[1][0])/L))
		NZ = int(np.round((bbox[2][1] - bbox[2][0])/L))
		NN = NX*NY*NZ
		S = makeStencil(hrad)
		NB = S.shape[0]
		Sf = (L*S).astype(np.float32)
		L0s = (np.sum((L*S)**2, axis=1)**0.5).astype(np.float32)
		jadd = (S[:,0] + NX*S[:,1] + NX*NY*S[:,2]).astype(np.int32)
		mi = np.sum(L0s)*L**3

		self.L = L
		self.NX = NX
		self.NY = NY
		self.NZ = NZ
		self.NN = NN
		self.NB = NB
		self.Sf = Sf
		self.L0s = L0s
		self.jadd = jadd
		self.mi = mi

class PDBoundaryConditions:
	def __init__(self, appu, ntau):
		self.appu = appu
		self.ntau = ntau

class PDModel:
	def __init__(self, mat, geom, bcs, dtype=np.float64):
		kappa = mat.E/(1-2*mat.nu)/3
		mu = mat.E/(1+mat.nu)/2
		cn = mat.tsi*mat.rho*np.pi/geom.L*(mat.E/mat.rho/3/(1-2*mat.nu))**0.5
		dt = geom.L / np.pi * (mat.rho/mat.E*3*(1-2*mat.nu))**0.5 * ((1+mat.tsi**2)**0.5 - mat.tsi)

		source = open("modules\\kernels.cu", "r")
		src = source.read()
		src = src.replace("NBr",str(geom.NB))
		src = src.replace("NDr",str(3))
		src = src.replace("NNr",str(geom.NN))
		src = src.replace("mur",str(mu))
		src = src.replace("NXr",str(geom.NX))
		src = src.replace("NYr",str(geom.NY))
		src = src.replace("NZr",str(geom.NZ))
		src = src.replace("Lr",str(geom.L))
		src = src.replace("dtr",str(dt))
		src = src.replace("appur",str(bcs.appu))
		src = src.replace("ntaur",str(bcs.ntau))
		src = src.replace("rhor",str(mat.rho))
		src = src.replace("cnr",str(cn))
		src = src.replace("ecritr",str(mat.ecrit))
		src = src.replace("dlmltr",str(geom.L**6/geom.mi/geom.mi*(9*kappa - 15*mu)))
		src = src.replace("fmltr",str(2*15*mu/geom.mi*geom.L**3))
		src = src.replace("SHr",str(4*geom.NB + 1))
		src = src.replace("L0s[]","L0s["+str(geom.NB)+"] = "+np.array2string(geom.L0s,separator = ',',max_line_width=np.nan).replace('[','{').replace(']','}'))
		src = src.replace("jadd[]","jadd["+str(geom.NB)+"] = "+np.array2string(geom.jadd,separator = ',',max_line_width=np.nan).replace('[','{').replace(']','}'))

		dsize = 8
		if dtype == np.float32:
			dsize = 4
			src = src.replace("double","float")
			src = src.replace("sqrt","sqrtf")
		self.dtype = dtype
		mod = SourceModule(src, options=["--use_fast_math"])

		self.d_dil = cuda.mem_alloc(geom.NN*dsize)
		self.d_u = cuda.mem_alloc(3*geom.NN*dsize)
		self.d_du = cuda.mem_alloc(3*geom.NN*dsize)
		self.d_ddu = cuda.mem_alloc(3*geom.NN*dsize)
		self.d_Sf = cuda.mem_alloc_like(geom.Sf)
		self.d_dmg = cuda.mem_alloc(((geom.NB)*geom.NN + 7)//8)
		cuda.memcpy_htod(self.d_Sf, geom.Sf)

		self.d_calcForceState = mod.get_function("calcForceState")
		self.d_calcDilation = mod.get_function("calcDilation")

		# d_calcForceState.set_cache_config(cuda.func_cache.PREFER_L1)
		# d_calcDilation.set_cache_config(cuda.func_cache.PREFER_L1)

		self.geom = geom

	def solve(self, NT, TPB = 128):
		NN = self.geom.NN
		NB = self.geom.NB
		d_Sf = self.d_Sf
		d_dil = self.d_dil
		d_u = self.d_u
		d_du = self.d_du
		d_ddu = self.d_ddu
		d_dmg = self.d_dmg
		BPG = (NN+TPB-1)//TPB

		# print("Begining simulation: ",NN)

		# t0 = time.time()
		for tt in range(NT):
			self.d_calcDilation(d_Sf, d_u, d_dil, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
			self.d_calcForceState(d_Sf, d_dil, d_u, d_du, d_ddu, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
		pycuda.driver.Context.synchronize()
		# tm = (time.time()-t0)
		# print(NN, tt, tm, tm/(tt+1))

	def get_displacement(self):
		NN = self.geom.NN
		u = np.empty(3*NN,dtype=self.dtype)
		cuda.memcpy_dtoh(u, self.d_u)
		return u[:NN], u[NN:2*NN], u[2*NN:] 

	def get_coords(self):
		inds = np.arange(self.geom.NN)
		z = self.geom.L*(inds//(self.geom.NX*self.geom.NY) + .5)
		y = self.geom.L*(inds%(self.geom.NX*self.geom.NY)//self.geom.NX + .5)
		x = self.geom.L*(inds%self.geom.NX + .5)
		return x, y, z

