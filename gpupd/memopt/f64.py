import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time
# from matplotlib import pyplot as plt
# import matplotlib.animation as anim
import math
import numpy as np
import sys


NX = int(sys.argv[1])
bbox = [[0, 1], [0, 1], [0, 1]] # Bounding box for geometry
E = np.float64(2e11)
nu = np.float64(0.3)
hrad = np.float64(3.01)
ntau = np.float64(500)
appu = np.float64(0.1)
rho = np.float64(1250.)
tsi = np.float64(1)
ecrit = np.float64(10000)
TPB = 128

L = np.float64((bbox[0][1] - bbox[0][0])/NX)
NY = int(np.round((bbox[1][1] - bbox[1][0])/L))
NZ = int(np.round((bbox[2][1] - bbox[2][0])/L))
NN = NX*NY*NZ
L3 = np.float64(L**3)
kappa = np.float64(E/(1-2*nu)/3)
mu = np.float64(E/(1+nu)/2)
cn = np.float64(tsi*rho*np.pi/L*(E/rho/3/(1-2*nu))**0.5)
dt = np.float64(L / np.pi * (rho/E*3*(1-2*nu))**0.5 * ((1+tsi**2)**0.5 - tsi) )
BPG = (NN+TPB-1)//TPB

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

# Make Geometry
S = makeStencil(hrad)
NB = S.shape[0]
Sf = (L*S).astype(np.float32)
L0s = (np.sum((L*S)**2, axis=1)**0.5).astype(np.float64)
jadd = (S[:,0] + NX*S[:,1] + NX*NY*S[:,2]).astype(np.int32)
mi = np.sum(L0s)*L3

mod = SourceModule("""
    #include <math.h>
    #include <inttypes.h>

    __device__ __constant__ float L = """+str(L)+""";
    __device__ __constant__ int NX = """+str(NX)+""";
    __device__ __constant__ int NY = """+str(NY)+""";
    __device__ __constant__ int NZ = """+str(NZ)+""";
    __device__ __constant__ int NB = """+str(NB)+""";
    __device__ __constant__ int NN = """+str(NN)+""";
    __device__ __constant__ double L0s["""+str(NB)+"""] = """+np.array2string(L0s,separator = ',',max_line_width=np.nan).replace('[','{').replace(']','}')+""";
    __device__ __constant__ int jadd["""+str(NB)+"""] = """+np.array2string(jadd,separator = ',',max_line_width=np.nan).replace('[','{').replace(']','}')+""";
    __device__ __constant__ double dt = """+str(dt)+""";
    __device__ __constant__ double appu = """+str(appu)+""";
    __device__ __constant__ double ntau = """+str(ntau)+""";
    __device__ __constant__ double rho = """+str(rho)+""";
    __device__ __constant__ double cn = """+str(cn)+""";
    __device__ __constant__ double ecrit = """+str(ecrit)+""";

    __device__ __constant__ double dlmlt = """+str(L3*L3/mi/mi*(9*kappa - 15*mu))+""";
    __device__ __constant__ double fmlt = """+str(2*15*mu/mi*L3)+""";

    __device__ __constant__ float ZERf = 0;
    __device__ __constant__ float ONEf = 1;
    __device__ __constant__ float HLF = 0.5;
    __device__ __constant__ float TRE = 3;

    __device__ __constant__ double ZER = 0;
    __device__ __constant__ double ONE = 1;
    __device__ __constant__ double TWO = 2;

    __device__ int tt = 0;

    __device__ __shared__ float sh_Sf["""+str(4*NB + 1)+"""];

    __device__ bool Chi(float x, float y, float z){
        return (x>=ZERf) && (x<=ONEf) && (y>=ZERf) && (y<=ONEf) && (z>=ZERf) && (z<=ONEf);
    }

    __device__ bool TestBit(bool A[],  int64_t k ){
        return ( (A[k/8] & (1 << (k%8) )) != 0 ) ;     
    }

    __device__ void  SetBit(bool A[],  int64_t k ){
        A[k/8] |= 1 << (k%8);
    }

    __global__ void calcDilation(float *d_Sf, double *d_u, double *d_dil, bool *d_dmg){
        int tix = threadIdx.x;
        int iind = blockIdx.x * blockDim.x + tix;

        int k = iind/(NX*NY);
        int j = iind%(NX*NY)/NX;
        int i = iind%NX;

        if(iind==0){
            tt += 1;
        }

        if(tix<NB){
            sh_Sf[tix] = d_Sf[tix];
            sh_Sf[NB+tix] = d_Sf[NB+tix];
            sh_Sf[2*NB+tix] = d_Sf[2*NB+tix];
        }
        __syncthreads();

        float xi = L*(i+HLF);
        float yi = L*(j+HLF);
        float zi = L*(k+HLF);

        if (Chi(xi,yi,zi)) {
            double ui = d_u[iind];
            double vi = d_u[NN+iind];
            double wi = d_u[2*NN+iind];

            double dil = ZER;
            for (int64_t b = 0;b<NB;b++){
                float dx2 = sh_Sf[b];
                float dy2 = sh_Sf[NB+b];
                float dz2 = sh_Sf[2*NB+b];
                if (Chi(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, b*NN + iind)) {
                    int jind = iind + jadd[b];
                    double uj = d_u[jind];
                    double vj = d_u[NN+jind];
                    double wj = d_u[2*NN+jind];
                    double L0 = L0s[b];
                    double A = dx2+uj-ui;
                    double B = dy2+vj-vi;
                    double C = dz2+wj-wi;
                    double LN = sqrt(A*A + B*B + C*C);
                    double eij = LN - L0;
                    if (eij/L0 > ecrit){
                        SetBit(d_dmg, b*NN + iind);
                        printf("Bond broke");
                    }else{
                        dil += eij/L0;
                    }
                }
            }
            d_dil[iind] = dil*dlmlt;
        }
    }

    __global__ void calcForceState(float *d_Sf, double *d_dil, double *d_u, double *d_du, double *d_ddu, bool *d_dmg) {
        int tix = threadIdx.x;
        int iind = blockIdx.x * blockDim.x + tix;

        int k = iind/(NX*NY);
        int j = iind%(NX*NY)/NX;
        int i = iind%NX;

        if(tix<NB){
            sh_Sf[tix] = d_Sf[tix];
            sh_Sf[NB+tix] = d_Sf[NB+tix];
            sh_Sf[2*NB+tix] = d_Sf[2*NB+tix];
        }
        __syncthreads();

        float xi = L*(i+HLF);
        float yi = L*(j+HLF);
        float zi = L*(k+HLF);

        if (Chi(xi,yi,zi)) {
            double ui = d_u[iind];
            double vi = d_u[NN+iind];
            double wi = d_u[2*NN+iind];
            double dili = d_dil[iind];

            double pFx = ZER;
            double pFy = ZER;
            double pFz = ZER;

            for (int64_t b = 0;b<NB;b++){
                float dx2 = sh_Sf[b];
                float dy2 = sh_Sf[NB+b];
                float dz2 = sh_Sf[2*NB+b];
                if (Chi(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, b*NN + iind)) {
                    int jind = iind + jadd[b];
                    double uj = d_u[jind];
                    double vj = d_u[NN+jind];
                    double wj = d_u[2*NN+jind];
                    double dilj = d_dil[jind];
                    double L0 = L0s[b];
                    double A = dx2+uj-ui;
                    double B = dy2+vj-vi;
                    double C = dz2+wj-wi;
                    double LN = sqrt(A*A + B*B + C*C);
                    double eij = LN - L0;
                    double fsm = dili + dilj + eij/L0*fmlt;
                    double dln = fsm/LN;
                    pFx += dln*A;
                    pFy += dln*B;
                    pFz += dln*C;
                }
            }

            double du = d_du[iind];
            double dv = d_du[NN+iind];
            double dw = d_du[2*NN+iind];
            double ddu = d_ddu[iind];
            double ddv = d_ddu[NN+iind];
            double ddw = d_ddu[2*NN+iind];

            double vhx = du + dt/TWO*ddu;
            double vhy = dv + dt/TWO*ddv;
            double vhz = dw + dt/TWO*ddw;
            double ddun = (pFx - cn*vhx)/rho;
            double ddvn = (pFy - cn*vhy)/rho;
            double ddwn = (pFz - cn*vhz)/rho;

            if (xi < TRE*L || xi>1-TRE*L){
                d_u[iind] = appu*(xi-3*L)*min(ONE,tt/ntau);
            }else{
                d_u[iind] = ui + dt*du + dt*dt/TWO*ddu;
                d_du[iind] = du + dt/TWO*(ddu + ddun);
                d_ddu[iind] = ddun;
            }
            if (yi < TRE*L){
                d_u[NN+iind] = ZER;
            }else{
                d_ddu[NN+iind] = ddvn;
                d_du[NN+iind] = dv + dt/TWO*(ddv + ddvn);
                d_u[NN+iind] = vi + dt*dv + dt*dt/TWO*ddv;
            }
            if (zi < TRE*L){
                d_u[2*NN+iind] = ZER;
            }else{
                d_ddu[2*NN+iind] = ddwn;
                d_du[2*NN+iind] = dw + dt/TWO*(ddw + ddwn);
                d_u[2*NN+iind] = wi + dt*dw + dt*dt/TWO*ddw;
            }
        }
    }
    """, options=["--use_fast_math"])

d_dil = cuda.mem_alloc(NN*L.dtype.itemsize)
d_u = cuda.mem_alloc(3*NN*L.dtype.itemsize)
d_du = cuda.mem_alloc(3*NN*L.dtype.itemsize)
d_ddu = cuda.mem_alloc(3*NN*L.dtype.itemsize)
d_Sf = cuda.mem_alloc_like(Sf)
d_dmg = cuda.mem_alloc(((NB)*NN + 7)//8)
cuda.memcpy_htod(d_Sf, Sf)

d_calcForceState = mod.get_function("calcForceState")
d_calcDilation = mod.get_function("calcDilation")

d_calcForceState.set_cache_config(cuda.func_cache.PREFER_L1)
d_calcDilation.set_cache_config(cuda.func_cache.PREFER_L1)
dil = np.empty((NN))

print("Begining simulation: ",NN)

t0 = time.time()
for tt in range(100):
    d_calcDilation(d_Sf, d_u, d_dil, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
    d_calcForceState(d_Sf, d_dil, d_u, d_du, d_ddu, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = (4*NB + 1)*4)
pycuda.driver.Context.synchronize()
tm = (time.time()-t0)
print(NN, tt, tm, tm/(tt+1))



# inds = np.arange(NN)
# i = (inds%NX)
# j = (inds%(NX*NY)//NX)
# k = (inds//(NX*NY))
# xs = L*i + L/2
# ys = L*j + L/2
# zs = L*k + L/2
# fspc = 4
# M = 1
# filt = (i&fspc==0) & (j%fspc==0) & (k%fspc==0)#np.ones(NN)>0#
# tt = 0
# u = np.empty(3*NN,dtype=np.float64)
# def update_graph(n):
#     global tt
#     for _ in range(100):
#         d_calcDilation(d_Sf, d_u, d_dil, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = 0)
#         d_calcForceState(d_Sf, d_dil, d_u, d_du, d_ddu, d_dmg, block = (TPB, 1, 1), grid = (BPG, 1 , 1), shared = 0)

#     cuda.memcpy_dtoh(u, d_u)
#     graph._offsets3d = (xs[filt]+M*u[:NN][filt], ys[filt]+M*u[NN:2*NN][filt], zs[filt]+M*u[2*NN:][filt])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(xs[filt]+M*appu*(xs[filt]-3*L), ys[filt]-M*appu*nu*(ys[filt]-3*L), zs[filt]-M*appu*nu*(zs[filt]-3*L),'x')
# graph = ax.scatter(xs[filt], ys[filt], zs[filt])
# ani = anim.FuncAnimation(fig, update_graph)
# plt.show()