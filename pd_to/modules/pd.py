import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import pycuda.reduction as reduction
import pycuda.gpuarray as gpuarray


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
		self.hrad = hrad
		self.bbox = bbox

class PDBoundaryConditions:
	def __init__(self, geom, ntau = 500):
		self.ntau = ntau
		NN = geom.NN
		NX = geom.NX
		NY = geom.NY
		L = geom.L
		inds = np.arange(NN)
		i = (inds%NX)
		j = (inds%(NX*NY)//NX)
		k = (inds//(NX*NY))
		self.x = L*i + L/2 + geom.bbox[0][0]
		self.y = L*j + L/2 + geom.bbox[1][0]
		self.z = L*k + L/2 + geom.bbox[2][0]
		self.NBCi = np.zeros(NN, dtype = np.int32) - 1
		self.NBC = []
		self.EBCi = np.zeros((3, NN), dtype = np.int32) - 1
		self.EBC = []
		self.geom = geom

	def makeFilt(self, bbox):
		return (self.x>bbox[0][0]) & (self.x<bbox[0][1]) & \
			   (self.y>bbox[1][0]) & (self.y<bbox[1][1]) & \
			   (self.z>bbox[2][0]) & (self.z<bbox[2][1])

	def addForce(self, bbox, vec):
		filt = self.makeFilt(bbox)
		self.NBCi[filt] = len(self.NBC)
		self.NBC.append(vec)

	def addDistributedForce(self, bbox, vec):
		filt = self.makeFilt(bbox)
		self.NBCi[filt] = len(self.NBC)
		self.NBC.append(vec/np.sum(filt)/self.geom.L**3)

	def addFixed(self, bbox, val, dims):
		filt = self.makeFilt(bbox)
		for d in dims:
			self.EBCi[d, filt] = len(self.EBC)
		self.EBC.append(val)

	def addFixedFunctional(self, bbox, func):
		filt = self.makeFilt(bbox)
		vals = func(self.x[filt], self.y[filt], self.z[filt])
		for d in range(3):
			self.EBCi[d, filt] = len(self.EBC) + np.arange(len(vals[d]))
			[self.EBC.append(val) for val in vals[d]]


class PDOptimizer:
	def __init__(self, alpha, volfrac, penal, tol=0.000001):
		self.alpha = alpha
		self.volfrac = volfrac
		self.tol = tol
		self.penal = penal

	def step(self, pd):
		RM = self.volfrac
		d_Wt = pd.sum_reduce(pd.d_W)
		while RM > self.tol:
			cuda.memcpy_htod(self.d_RM, np.array([RM], dtype = pd.dtype))
			self.d_calcKbar(pd.d_Sf, d_Wt, self.d_RM, pd.d_W, pd.d_dmg, self.d_kbar, block = (pd.TPB, 1, 1), grid = ((pd.geom.NN+pd.TPB-1)//pd.TPB, 1 , 1))
			AM = pd.sum_reduce(self.d_kbar).get()/pd.geom.NN
			RM = self.volfrac - AM
		self.d_updateK(pd.d_k, self.d_kbar, pd.d_NBCi, pd.d_EBCi, block = (pd.TPB, 1, 1), grid = ((pd.geom.NN+pd.TPB-1)//pd.TPB, 1 , 1))
		k = pd.d_k.get()
		kb = self.d_kbar.get()

class PDModel:
	def __init__(self, mat, geom, bcs, opt=None, dtype=np.float64, TPB = 128):
		kappa = mat.E/(1-2*mat.nu)/3
		mu = mat.E/(1+mat.nu)/2
		am = 15*mu
		cn = mat.tsi*mat.rho*np.pi/geom.L*(mat.E/mat.rho/3/(1-2*mat.nu))**0.5
		dt = 1
		bc = 12*mat.E/np.pi/geom.hrad**4
		if opt is None:
			alpha = 0
		else:
			alpha = opt.alpha

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
		src = src.replace("ntaur",str(bcs.ntau))
		src = src.replace("rhor",str(mat.rho))
		src = src.replace("cnr",str(cn))
		src = src.replace("ecritr",str(mat.ecrit))
		src = src.replace("dlmltr",str(geom.L**6/geom.mi/geom.mi*(9*kappa - 15*mu)))
		src = src.replace("fmltr",str(2*15*mu/geom.mi*geom.L**3))
		src = src.replace("mvecr",str(.25*dt**2 * 4/3*np.pi*geom.hrad**3 * bc / geom.L **2))
		src = src.replace("SHr",str(4*geom.NB + 1))
		src = src.replace("TPB",str(TPB))
		src = src.replace("L0s[]","L0s["+str(geom.NB)+"] = "+np.array2string(geom.L0s,separator = ',',max_line_width=np.nan).replace('[','{').replace(']','}'))
		src = src.replace("jadd[]","jadd["+str(geom.NB)+"] = "+np.array2string(geom.jadd,separator = ',',max_line_width=np.nan).replace('[','{').replace(']','}'))
		src = src.replace("alphar",str(alpha))
		src = src.replace("amr",str(am))
		src = src.replace("kappar",str(kappa))
		src = src.replace("hradr",str(geom.L*geom.hrad))
		src = src.replace("xlr",str(geom.bbox[0][0]))
		src = src.replace("xhr",str(geom.bbox[0][1]))
		src = src.replace("ylr",str(geom.bbox[1][0]))
		src = src.replace("yhr",str(geom.bbox[1][1]))
		src = src.replace("zlr",str(geom.bbox[2][0]))
		src = src.replace("zhr",str(geom.bbox[2][1]))
		if opt is None:
			src = src.replace("penalr",str(0))
		else:
			src = src.replace("penalr",str(opt.penal))

		dsize = 8
		if dtype == np.float32:
			dsize = 4
			src = src.replace("double","float")
			src = src.replace("sqrt","sqrtf")
		self.dtype = dtype
		mod = SourceModule(src, options=["--use_fast_math"])

		self.d_m = gpuarray.GPUArray([geom.NN], dtype)
		self.d_dil = gpuarray.GPUArray([geom.NN], dtype)
		self.d_u = gpuarray.GPUArray([3*geom.NN], dtype)
		self.d_up = gpuarray.GPUArray([3*geom.NN], dtype)
		self.d_F = gpuarray.GPUArray([3*geom.NN], dtype)
		self.d_cd = gpuarray.GPUArray([geom.NN], dtype)
		self.d_cn = gpuarray.GPUArray([geom.NN], dtype)
		self.d_Sf = gpuarray.to_gpu(geom.Sf)
		self.d_dmg = gpuarray.GPUArray([((geom.NB)*geom.NN + 7)//8], np.bool_)
		self.d_NBCi = gpuarray.to_gpu(bcs.NBCi)
		self.d_EBCi = gpuarray.to_gpu(bcs.EBCi.flatten())
		self.d_NBC = gpuarray.to_gpu(np.array(bcs.NBC).astype(np.float32).flatten())
		self.d_EBC = gpuarray.to_gpu(np.array(bcs.EBC).astype(np.float32))

		self.d_c = cuda.mem_alloc(dsize)

		self.d_k = gpuarray.to_gpu(np.ones(geom.NN, dtype=dtype))
		self.d_W = gpuarray.GPUArray([geom.NN], dtype)
		self.d_Ft = gpuarray.GPUArray([geom.NN], dtype)
		
		self.d_calcVolume = mod.get_function("calcVolume")
		self.d_calcForce = mod.get_function("calcForce")
		self.d_calcDilation = mod.get_function("calcDilation")
		self.d_calcDisplacement = mod.get_function("calcDisplacement")
		self.sum_reduce = reduction.get_sum_kernel(dtype, dtype)
		# self.d_calcForce.set_cache_config(cuda.func_cache.PREFER_L1)
		# self.d_calcDilation.set_cache_config(cuda.func_cache.PREFER_L1)

		if opt is not None:
			opt.d_kbar = gpuarray.GPUArray([geom.NN], dtype)
			opt.d_RM = cuda.mem_alloc(dsize)
			opt.d_calcKbar = mod.get_function("calcKbar")
			opt.d_updateK = mod.get_function("updateK")

		self.TPB = TPB
		self.geom = geom
		self.opt = opt
		self.bcs = bcs
		self.ft = 0

	def evalC(self):
		cn1 = self.sum_reduce(self.d_cn).get()
		cn2 = self.sum_reduce(self.d_cd).get()
		if (cn2 != 0.0):
			if ((cn1 / cn2) > 0.0):
				cn = 2.0 * (cn1 / cn2)**0.5
			else:
				cn = 0.0
		else:
			cn = 0.0
		if (cn > 2.0):
			cn = 1.9
		return np.array([cn], dtype = self.dtype)

	def solve(self, NT):
		NN = self.geom.NN
		NB = self.geom.NB
		TPB = self.TPB
		d_Sf = self.d_Sf
		d_dil = self.d_dil
		d_u = self.d_u
		d_up = self.d_up
		d_F = self.d_F
		d_cn = self.d_cn
		d_cd = self.d_cd
		d_c = self.d_c
		d_dmg = self.d_dmg
		d_NBC = self.d_NBC
		d_EBC = self.d_EBC
		d_NBCi = self.d_NBCi
		d_EBCi = self.d_EBCi
		d_W = self.d_W
		d_k = self.d_k
		d_Ft = self.d_Ft
		d_m = self.d_m
		BPG = (NN+TPB-1)//TPB

		self.d_calcVolume(d_Sf, d_m, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
		
		for tt in range(NT):
			self.d_calcDilation(d_Sf, d_u, d_dil, d_dmg, d_W, d_m, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
			self.d_calcForce(d_Sf, d_dil, d_u, d_dmg, d_F, d_up, d_cd, d_cn, d_EBCi, d_k, d_W, d_Ft, d_m, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
			cuda.memcpy_htod(d_c, self.evalC())
			self.d_calcDisplacement(d_c, d_u, d_up, d_F, d_NBCi, d_NBC, d_EBCi, d_EBC, block = (TPB, 1, 1), grid = (BPG, 1 , 1))
			if tt%50==0:
				ft = self.sum_reduce(d_Ft).get()
				print(tt, ft, self.ft)
				if ft>self.ft:
					self.ft = ft
				elif ft<0.0001*self.ft:
					break

		if self.opt is not None:
			self.opt.step(self)

	def get_displacement(self):
		NN = self.geom.NN
		u = self.d_u.get()
		return u[:NN], u[NN:2*NN], u[2*NN:] 
	
	def get_fill(self):
		return self.d_k.get()

	def get_W(self):
		return self.d_W.get()

	def get_coords(self):
		inds = np.arange(self.geom.NN)
		z = self.geom.L*(inds//(self.geom.NX*self.geom.NY) + .5)
		y = self.geom.L*(inds%(self.geom.NX*self.geom.NY)//self.geom.NX + .5)
		x = self.geom.L*(inds%self.geom.NX + .5)
		return x, y, z

