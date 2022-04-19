#include <math.h>
#include <inttypes.h>


__device__ __constant__ float L = Lr;
__device__ __constant__ int NX = NXr;
__device__ __constant__ int NY = NYr;
__device__ __constant__ int NZ = NZr;
__device__ __constant__ int NB = NBr;
__device__ __constant__ int NN = NNr;
__device__ __constant__ double L0s[];
__device__ __constant__ int jadd[];
__device__ __constant__ double dt = dtr;
__device__ __constant__ double appu = appur;
__device__ __constant__ double ntau = ntaur;
__device__ __constant__ double rho = rhor;
__device__ __constant__ double cn = cnr;
__device__ __constant__ double ecrit = ecritr;

__device__ __constant__ double dlmlt = dlmltr;
__device__ __constant__ double fmlt = fmltr;

__device__ __constant__ float ZERf = 0;
__device__ __constant__ float ONEf = 1;
__device__ __constant__ float HLF = 0.5;
__device__ __constant__ float TRE = 3;

__device__ __constant__ double ZER = 0;
__device__ __constant__ double ONE = 1;
__device__ __constant__ double TWO = 2;

__device__ int tt = 0;

__device__ __shared__ float sh_Sf[SHr]; // 4*NB + 1

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