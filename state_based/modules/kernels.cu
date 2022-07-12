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
__device__ __constant__ double ntau = ntaur;
__device__ __constant__ double rho = rhor;
__device__ __constant__ double ecrit = ecritr;
__device__ __constant__ double bc = bcr;
__device__ __constant__ double kappa = kappar;
__device__ __constant__ double am = amr;
__device__ __constant__ double dV = dVr;
__device__ __constant__ double mass = massr;
__device__ __constant__ double MLT = MLTr;

__device__ __constant__ float xh = xhr;
__device__ __constant__ float xl = xlr;
__device__ __constant__ float yh = yhr;
__device__ __constant__ float yl = ylr;
__device__ __constant__ float zh = zhr;
__device__ __constant__ float zl = zlr;

__device__ __constant__ int penal = penalr;
__device__ __constant__ double alpha = alphar;
__device__ __constant__ double hrad = hradr;

__device__ __constant__ float HLF = 0.5;
__device__ __constant__ float TRE = 3;
__device__ __constant__ double ZER = 0;
__device__ __constant__ double ONE = 1;
__device__ __constant__ double TWO = 2;

__device__ int tt = 0;

__device__ __shared__ float sh_Sf[SHr]; // 4*NB + 1

__device__ bool TestBbox(float x, float y, float z){
    return (x>=xl) && (x<=xh) && (y>=yl) && (y<=yh) && (z>=zl) && (z<=zh);
}

__device__ bool TestBit(bool A[],  int64_t k ){
    return ( (A[k/8] & (1 << (k%8) )) != 0 ) ;     
}

__device__ void  SetBit(bool A[],  int64_t k ){
    A[k/8] |= 1 << (k%8);
}

__global__ void calcVolume(float *d_Sf, double *d_m, bool *d_dmg, bool *d_chi){
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

    float xi = L*(i+HLF) + xl;
    float yi = L*(j+HLF) + yl;
    float zi = L*(k+HLF) + zl;

    if (TestBbox(xi,yi,zi) && d_chi[iind]) {
        double mi = ZER;
        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, b*NN + iind)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double L0 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
                    mi += L0*L*L*L;
                }
            }
        }
        d_m[iind] = mi;
    }
}

__global__ void calcDilation(float *d_Sf, double *d_u, double *d_dil, bool *d_dmg, double *d_m, bool *d_chi){
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

    float xi = L*(i+HLF) + xl;
    float yi = L*(j+HLF) + yl;
    float zi = L*(k+HLF) + zl;

    if (TestBbox(xi,yi,zi) && d_chi[iind]) {
        double ui = d_u[iind];
        double vi = d_u[NN+iind];
        double wi = d_u[2*NN+iind];

        double mi = d_m[iind];

        double dil = ZER;
        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, b*NN + iind)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double uj = d_u[jind];
                    double vj = d_u[NN+jind];
                    double wj = d_u[2*NN+jind];
                    double L0 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);

                    double A = dx2+uj-ui;
                    double B = dy2+vj-vi;
                    double C = dz2+wj-wi;
                    double LN = sqrt(A*A + B*B + C*C);
                    double eij = LN - L0;
                    if (eij/L0 > ecrit){
                        SetBit(d_dmg, b*NN + iind);
                        printf("Bond broken");
                    }else{
                        dil += eij*L*L*L;
                    }
                }
            }
        }
        d_dil[iind] = dil*3/mi;
    }
}

__global__ void calcForce(float *d_Sf, double *d_u, double *d_dil, double *d_m, bool *d_dmg, double *d_F, 
        double *d_vh, double *d_cd, double *d_cn, int *d_EBCi, double *d_k, double *d_W,
        double *d_Ft, bool *d_chi) {

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

    float xi = L*(i+HLF) + xl;
    float yi = L*(j+HLF) + yl;
    float zi = L*(k+HLF) + zl;

    if (TestBbox(xi,yi,zi) && d_chi[iind]) {
        double ui = d_u[iind];
        double vi = d_u[NN+iind];
        double wi = d_u[2*NN+iind];
        double ki = pow(d_k[iind],penal);
        double mi = d_m[iind];
        double dili = d_dil[iind];

        double fx = ZER;
        double fy = ZER;
        double fz = ZER;

        double Wsm = 0;

        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, b*NN + iind)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double uj = d_u[jind];
                    double vj = d_u[NN+jind];
                    double wj = d_u[2*NN+jind];
                    double L0 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);//L0s[b];
                    double A = dx2+uj-ui;
                    double B = dy2+vj-vi;
                    double C = dz2+wj-wi;
                    double LN = sqrt(A*A + B*B + C*C);
                    double eij = LN - L0;
                    double kj = pow(d_k[jind],penal);
                    double mj = d_m[jind];
                    double dilj = d_dil[jind];
                    double tij = ki*(3*kappa*dili + am*(eij/L0 - dili/3))/mi;
                    double tji = kj*(3*kappa*dilj + am*(eij/L0 - dilj/3))/mj;
                    double fsm = (tij + tji)*L*L*L;
                    fx += fsm*A/LN;
                    fy += fsm*B/LN;
                    fz += fsm*C/LN;
                    Wsm += 2*(ki*kj)/(ki + kj)*0.5*0.5*bc*eij*eij/L0*dV;
                }
            }
        }

        int ebcx = d_EBCi[iind];
        int ebcy = d_EBCi[NN + iind];
        int ebcz = d_EBCi[2*NN + iind];
        double vhx = d_vh[iind];
        double vhy = d_vh[NN + iind];
        double vhz = d_vh[2*NN + iind];
        double pfx = d_F[iind];
        double pfy = d_F[NN + iind];
        double pfz = d_F[2*NN + iind];
        double cn = ZER;
        double cd = ZER;
        if(ebcx<0 && vhx != ZER){
            cn -= MLT*ui*ui*(fx - pfx)/(ki*mass*dt*vhx);
        }
        cd += ui*ui;
        if(ebcy<0 && vhy != ZER){
            cn -= MLT*vi*vi*(fy - pfy)/(ki*mass*dt*vhy);
        }
        cd += vi*vi;
        if(ebcz<0 && vhz != ZER){
            cn -= MLT*wi*wi*(fz - pfz)/(ki*mass*dt*vhz);
        }
        cd += wi*wi;
        d_cn[iind] = cn;
        d_cd[iind] = cd;
        
        d_F[iind] = fx;
        d_F[NN + iind] = fy;
        d_F[2*NN + iind] = fz;
        
        if(ebcx<0 && ebcy<0 && ebcz<0){
            d_Ft[iind] = sqrt(fx*fx + fy*fy + fz*fz);
            d_W[iind] = Wsm;
        }
    }
}

__global__ void calcDisplacement(double *d_c, double *d_u, double *d_vh, double *d_F, int *d_NBCi, float *d_NBC, int *d_EBCi, float *d_EBC, double *d_k, bool *d_chi){
    int tix = threadIdx.x;
    int iind = blockIdx.x * blockDim.x + tix;
    int k = iind/(NX*NY);
    int j = iind%(NX*NY)/NX;
    int i = iind%NX;

    float xi = L*(i+HLF) + xl;
    float yi = L*(j+HLF) + yl;
    float zi = L*(k+HLF) + zl;
    double c = d_c[0];

    if (TestBbox(xi,yi,zi) && d_chi[iind]) {
        double pfx = d_F[iind];
        double pfy = d_F[NN + iind];
        double pfz = d_F[2*NN + iind];
        int nbci = d_NBCi[iind];
        if(nbci>=0){
            pfx += d_NBC[3*nbci];
            pfy += d_NBC[3*nbci + 1];
            pfz = d_NBC[3*nbci + 2];
        }
        double vhox = d_vh[iind];
        double vhoy = d_vh[NN + iind];
        double vhoz = d_vh[2*NN + iind];
        double ui = d_u[iind];
        double vi = d_u[NN + iind];
        double wi = d_u[2*NN + iind];
        double ki = d_k[iind];

        double vhx; double vhy; double vhz;

        if (tt==0){
            vhx = dt/max(0.05,ki)/mass * pfx / TWO;
            vhy = dt/max(0.05,ki)/mass * pfy / TWO;
            vhz = dt/max(0.05,ki)/mass * pfz / TWO;
        } else {
            vhx = ((TWO - c*dt)*vhox + TWO*dt/max(0.05,ki)/mass*pfx)/(TWO + c*dt);
            vhy = ((TWO - c*dt)*vhoy + TWO*dt/max(0.05,ki)/mass*pfy)/(TWO + c*dt);
            vhz = ((TWO - c*dt)*vhoz + TWO*dt/max(0.05,ki)/mass*pfz)/(TWO + c*dt);
        }
        int ebcx = d_EBCi[iind];
        int ebcy = d_EBCi[NN + iind];
        int ebcz = d_EBCi[2*NN + iind];
        if(ebcx<0){
            d_u[iind] = ui + dt*vhx;
        }else{
            d_u[iind] = d_EBC[ebcx]*min(ONE, tt/ntau);
        }
        if(ebcy<0){
            d_u[NN+iind] = vi + dt*vhy;
        }else{
            d_u[NN+iind] = d_EBC[ebcy]*min(ONE, tt/ntau);
        }
        if(ebcz<0){
            d_u[2*NN + iind] = wi + dt*vhz;
        }else{
            d_u[2*NN + iind] = d_EBC[ebcz]*min(ONE, tt/ntau);
        }
        d_vh[iind] = vhx;
        d_vh[NN + iind] = vhy;
        d_vh[2*NN + iind] = vhz;
    }
}

__global__ void calcKbar(float *d_Sf, double *d_Wt, double *d_RM, double *d_W, bool *d_dmg, double *d_kbar, bool *d_chi){
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

    float xi = L*(i+HLF) + xl;
    float yi = L*(j+HLF) + yl;
    float zi = L*(k+HLF) + zl;

    double Wt = d_Wt[0];
    double RM = d_RM[0];

    if (TestBbox(xi,yi,zi) && d_chi[iind]) {
        double kopti = d_W[iind] *NN * RM / Wt;
        double nsm = kopti*hrad;
        double dsm = hrad;
        for (int64_t b = 0; b<NB; b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, b*NN + iind)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double psi = max(ZER, hrad - L0s[b]);
                    nsm += psi * d_W[jind] * NN * RM / Wt;
                    dsm += psi;
                }
            }
        }
        d_kbar[iind] = max(0.0001, min(ONE, d_kbar[iind] + nsm / dsm));
    }
}

__global__ void updateK(double *d_k, double *d_kbar, int *d_NBCi, int *d_EBCi){
    int iind = blockIdx.x * blockDim.x + threadIdx.x;

    if(iind<NN){
        if(d_NBCi[iind]<0 && d_EBCi[iind]<0 && d_EBCi[NN + iind]<0 && d_EBCi[2*NN + iind]<0){
            d_k[iind] = alpha*d_k[iind] + (1 - alpha) * d_kbar[iind];
            d_kbar[iind] = ZER;
        }
    }
}


