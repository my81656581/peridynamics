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
__device__ __constant__ double ecrit = ecritr;
__device__ __constant__ double bc = bcr;
__device__ __constant__ double dV = dVr;
__device__ __constant__ double mass = massr;
__device__ __constant__ double rho = rhor;
__device__ __constant__ double MLT = MLTr;
__device__ __constant__ double emod = emodr;

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

__device__ bool TestBit(uint8_t *A,  int64_t k ){
    return ( (A[k/8] & (1 << (k%8) )) != 0 ) ;     
}

__device__ void  SetBit(uint8_t *A,  int64_t k ){
    A[k/8] |= 1 << (k%8);
}

__global__ void zeroT(){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        tt = 0;
    }
}

// __global__ void initCuts(float *d_Sf, bool *d_D, bool *d_chi, uint8_t *d_dmg){
//     int tix = threadIdx.x;
//     int iind = blockIdx.x * blockDim.x + tix;
//     int k = iind/(NX*NY);
//     int j = iind%(NX*NY)/NX;
//     int i = iind%NX;
//     if(tix<NB){
//         sh_Sf[tix] = d_Sf[tix];
//         sh_Sf[NB+tix] = d_Sf[NB+tix];
//         sh_Sf[2*NB+tix] = d_Sf[2*NB+tix];
//     }
//     __syncthreads();
//     float xi = L*(i+HLF) + xl;
//     float yi = L*(j+HLF) + yl;
//     float zi = L*(k+HLF) + zl;
//     if (TestBbox(xi,yi,zi) && d_chi[iind]) {
//         for (int64_t b = 0;b<NB;b++){
//             float dx2 = sh_Sf[b];
//             float dy2 = sh_Sf[NB+b];
//             float dz2 = sh_Sf[2*NB+b];
//             if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, NB*iind+b)) {
//                 int jind = iind + jadd[b];
//                 if(d_chi[jind]){
//                     for (int64_t b2 = 0;b2<NB;b2++){
//                         if(d_D[b*NB + b2] && !d_chi[iind + jadd[b2]]){
//                             SetBit(d_dmg, NB*iind+b);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

__global__ void initCuts(float *d_Sf, uint8_t *d_dmg){
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
    if (TestBbox(xi,yi,zi)) {
        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            float intsct = xi - dx2*yi/dy2;
            if (yi*(yi+dy2)<0 && intsct>=-.005-.01*L && intsct<=.005+.01*L){
                SetBit(d_dmg, NB*iind+b);      
            }
        }
    }
}

__global__ void setSCR(float *d_Sf, bool *d_chi, double *d_fncst){
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
    double Wx = 0;
    double Wy = 0;
    double Wz = 0;
    if (TestBbox(xi,yi,zi)) {
        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double L0 = sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2);
                    double fac = 0;
                    if (L0<=hrad-L/2){
                        fac = 1;
                    }else if(L0<=hrad+L/2){
                        fac = (hrad + L/2 - L0)/L;
                    }
                    double LX = sqrt(pow(dx2 + .001*(xi+dx2) - .001*xi, 2) + dy2*dy2 + dz2*dz2);
                    Wx += + 0.5 * 0.5 * bc * pow((LX - L0) / L0, 2) * L0 * dV * fac;
                    double LY = sqrt(pow(dy2 + .001*(yi+dy2) - .001*yi, 2) + dx2*dx2 + dz2*dz2);
                    Wy += + 0.5 * 0.5 * bc * pow((LY - L0) / L0, 2) * L0 * dV * fac;
                    double LZ = sqrt(pow(dx2 + .001*(zi+dz2) - .001*zi, 2) + dy2*dy2 + dx2*dx2);
                    Wz += + 0.5 * 0.5 * bc * pow((LZ - L0) / L0, 2) * L0 * dV * fac;
                }                
            }
        }
        d_fncst[iind] = 9.0 / 16.0 * emod * 1.0e-6 / Wx;
        d_fncst[NN + iind] = 9.0 / 16.0 * emod * 1.0e-6 / Wy;
        d_fncst[2*NN + iind] = 9.0 / 16.0 * emod * 1.0e-6 / Wz;
    }
}

__global__ void calcForceCorrected(float *d_Sf, double *d_u, uint8_t *d_dmg, double *d_F, 
        double *d_Ft, bool *d_chi, double *d_fncst) {

    int tix = threadIdx.x;
    int iind = blockIdx.x * blockDim.x + tix;

    if(iind==0){
        tt++;
    }

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

        double fx = ZER;
        double fy = ZER;
        double fz = ZER;

        float smA = 0;
        int smB = 0;

        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2)) {
                smB++;
                if (!TestBit(d_dmg, NB*iind+b)){
                    int jind = iind + jadd[b];
                    if(d_chi[jind]){
                        double uj = d_u[jind];
                        double vj = d_u[NN+jind];
                        double wj = d_u[2*NN+jind];
                        double L0 = sqrt(pow(dx2,2) + pow(dy2,2) + pow(dz2,2));//L0s[b];
                        double fac = 0;
                        if (L0<=hrad-L/2){
                            fac = 1;
                        }else if(L0<=hrad+L/2){
                            fac = (hrad + L/2 - L0)/L;
                        }
                        double scr;
                        if(abs(dz2)<1e-10){
                            double tht;
                            if (abs(dy2)<1e-10){
                                tht = 0;
                            }else if(abs(dx2)<1e-10){
                                tht = 1.57079326;
                            }else{
                                tht = atan(dy2/dx2);
                            }
                            double scx = (d_fncst[iind] + d_fncst[jind])/2;
                            double scy = (d_fncst[NN + iind] + d_fncst[NN + jind])/2;
                            scr = sqrt(1/(pow(cos(tht)/scx,2) + pow(sin(tht)/scy,2)));
                        }else if(abs(dx2)<1e-10 && abs(dy2)<1e-10){
                            scr = (d_fncst[2*NN + iind] + d_fncst[2*NN + jind])/2;
                        }else{
                            double tht = atan(dy2/dx2);
                            double phi = acos(dz2/L0);
                            double scx = (d_fncst[iind] + d_fncst[jind])/2;
                            double scy = (d_fncst[NN + iind] + d_fncst[NN + jind])/2;
                            double scz = (d_fncst[2*NN + iind] + d_fncst[2*NN + jind])/2;
                            scr = sqrt(1/(pow(cos(tht)*sin(phi)/scx,2) + pow(sin(tht)*sin(phi)/scy,2) + pow(cos(phi)/scz,2)));
                        }
                        double A = dx2+uj-ui;
                        double B = dy2+vj-vi;
                        double C = dz2+wj-wi;
                        double LN = sqrt(pow(A,2) + pow(B,2) + pow(C,2));
                        double eij = LN - L0;
                        double fsm = eij/L0*fac*scr;
                        fx += fsm*A/LN;
                        fy += fsm*B/LN;
                        fz += fsm*C/LN;
                        if (eij/L0 > ecrit && abs(yi)<.04/4){
                            SetBit(d_dmg, NB*iind+b);
                        }
                    }
                }else{
                    smA++;
                }
            }
        }

        d_Ft[iind] = 1 - smA/smB;

        fx *= bc*dV;
        fy *= bc*dV;
        fz *= bc*dV;
        
        d_F[iind] = fx;
        d_F[NN + iind] = fy;
        d_F[2*NN + iind] = fz;
    }
}

__global__ void calcForce(float *d_Sf, double *d_u, uint8_t *d_dmg, double *d_F, 
        double *d_vh, double *d_cd, double *d_cn, int *d_EBCi, double *d_k, double *d_W,
        double *d_Ft, bool *d_chi, int *d_NBCi) {

    int tix = threadIdx.x;
    int iind = blockIdx.x * blockDim.x + tix;

    if(iind==0){
        tt++;
    }

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

        double fx = ZER;
        double fy = ZER;
        double fz = ZER;
        double Wsm = ZER;

        for (int64_t b = 0;b<NB;b++){
            float dx2 = sh_Sf[b];
            float dy2 = sh_Sf[NB+b];
            float dz2 = sh_Sf[2*NB+b];
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, NB*iind+b)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double uj = d_u[jind];
                    double vj = d_u[NN+jind];
                    double wj = d_u[2*NN+jind];
                    double L0 = sqrt(pow(dx2,2) + pow(dy2,2) + pow(dz2,2));//L0s[b];                    
                    double A = dx2+uj-ui;
                    double B = dy2+vj-vi;
                    double C = dz2+wj-wi;
                    double LN = sqrt(pow(A,2) + pow(B,2) + pow(C,2));
                    double eij = LN - L0;
                    double kj = pow(d_k[jind],penal);
                    double fsm = TWO*(ki*kj)/(ki + kj)*eij/L0;
                    fx += fsm*A/LN;
                    fy += fsm*B/LN;
                    fz += fsm*C/LN;
                    Wsm += TWO*(ki*kj)/(ki + kj)*HLF*HLF*eij*eij/L0;
                    if (eij/L0 > ecrit){
                        SetBit(d_dmg, NB*iind+b);
                    }
                }
            }
        }

        fx *= bc*dV;
        fy *= bc*dV;
        fz *= bc*dV;
        Wsm *= bc*dV;    

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
        if(ebcx<ZER && vhx != ZER){
            cn -= MLT*ui*ui*(fx - pfx)/(ki*mass*dt*vhx);
        }
        cd += ui*ui;
        if(ebcy<ZER && vhy != ZER){
            cn -= MLT*vi*vi*(fy - pfy)/(ki*mass*dt*vhy);
        }
        cd += vi*vi;
        if(ebcz<ZER && vhz != ZER){
            cn -= MLT*wi*wi*(fz - pfz)/(ki*mass*dt*vhz);
        }
        cd += wi*wi;
        d_cn[iind] = cn;
        d_cd[iind] = cd;
        
        d_F[iind] = fx;
        d_F[NN + iind] = fy;
        d_F[2*NN + iind] = fz;
        
        if(ebcx<ZER && ebcy<ZER && ebcz<ZER && d_NBCi[iind]<ZER){
            d_Ft[iind] = abs(vhx) + abs(vhy) + abs(vhz);
            d_W[iind] = Wsm;
        }
    }
}

__global__ void calcDisplacement(double *d_c, double *d_u, double *d_vh, double *d_F, int *d_NBCi, float *d_NBC, int *d_EBCi, float *d_EBC, float *d_EBC0, double *d_k, bool *d_chi){
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
            pfx += d_NBC[3*nbci];//*min(ONE, tt/ntau);
            pfy += d_NBC[3*nbci + 1];
            pfz += d_NBC[3*nbci + 2];
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
            vhx = dt/ki/mass * pfx / TWO;
            vhy = dt/ki/mass * pfy / TWO;
            vhz = dt/ki/mass * pfz / TWO;
        } else {
            vhx = ((TWO - c*dt)*vhox + TWO*dt/ki/mass*pfx)/(TWO + c*dt);
            vhy = ((TWO - c*dt)*vhoy + TWO*dt/ki/mass*pfy)/(TWO + c*dt);
            vhz = ((TWO - c*dt)*vhoz + TWO*dt/ki/mass*pfz)/(TWO + c*dt);
        }
        int ebcx = d_EBCi[iind];
        int ebcy = d_EBCi[NN + iind];
        int ebcz = d_EBCi[2*NN + iind];
        if(ebcx<0){
            d_u[iind] = ui + dt*vhx;
        }else{
            float e0 = d_EBC0[ebcx];
            d_u[iind] = e0 + (d_EBC[ebcx] - e0)*min(ONE, tt/ntau);
        }
        if(ebcy<0){
            d_u[NN+iind] = vi + dt*vhy;
        }else{
            float e0 = d_EBC0[ebcy];
            d_u[NN+iind] = e0 + (d_EBC[ebcy] - e0)*min(ONE, tt/ntau);
        }
        if(ebcz<0){
            d_u[2*NN + iind] = wi + dt*vhz;
        }else{
            float e0 = d_EBC0[ebcz];
            d_u[2*NN + iind] = e0 + (d_EBC[ebcz] - e0)*min(ONE, tt/ntau);
        }
        d_vh[iind] = vhx;
        d_vh[NN + iind] = vhy;
        d_vh[2*NN + iind] = vhz;
    }
}

__global__ void calcDisplacementEuler(double *d_u, double *d_vh, double *d_F, int *d_NBCi, float *d_NBC, int *d_EBCi, float *d_EBC, float *d_EBC0, bool *d_chi){
    int tix = threadIdx.x;
    int iind = blockIdx.x * blockDim.x + tix;
    int k = iind/(NX*NY);
    int j = iind%(NX*NY)/NX;
    int i = iind%NX;
    float xi = L*(i+HLF) + xl;
    float yi = L*(j+HLF) + yl;
    float zi = L*(k+HLF) + zl;

    if (TestBbox(xi,yi,zi) && d_chi[iind]) {
        double pfx = d_F[iind];
        double pfy = d_F[NN + iind];
        double pfz = d_F[2*NN + iind];
        int nbci = d_NBCi[iind];
        if(nbci>=0){
            pfx += d_NBC[3*nbci];
            pfy += d_NBC[3*nbci + 1];
            pfz += d_NBC[3*nbci + 2];
        }
        double vhox = d_vh[iind];
        double vhoy = d_vh[NN + iind];
        double vhoz = d_vh[2*NN + iind];
        double ui = d_u[iind];
        double vi = d_u[NN + iind];
        double wi = d_u[2*NN + iind];
        double vhx; double vhy; double vhz;
        vhx = vhox + pfx/rho*dt;
        vhy = vhoy + pfy/rho*dt;
        vhz = vhoz + pfz/rho*dt;
        int ebcx = d_EBCi[iind];
        int ebcy = d_EBCi[NN + iind];
        int ebcz = d_EBCi[2*NN + iind];
        if(ebcx<0){
            d_u[iind] = ui + dt*vhx;
        }else{
            float e0 = d_EBC0[ebcx];
            d_u[iind] = e0 + (d_EBC[ebcx] - e0)*min(ONE, tt/ntau);
        }
        if(ebcy<0){
            d_u[NN+iind] = vi + dt*vhy;
        }else{
            float e0 = d_EBC0[ebcy];
            d_u[NN+iind] = e0 + (d_EBC[ebcy] - e0)*min(ONE, tt/ntau);
        }
        if(ebcz<0){
            d_u[2*NN + iind] = wi + dt*vhz;
        }else{
            float e0 = d_EBC0[ebcz];
            d_u[2*NN + iind] = e0 + (d_EBC[ebcz] - e0)*min(ONE, tt/ntau);
        }
        d_vh[iind] = vhx;
        d_vh[NN + iind] = vhy;
        d_vh[2*NN + iind] = vhz;
    }
}

__global__ void calcKbar(float *d_Sf, double *d_Wt, double *d_RM, double *d_W, uint8_t *d_dmg, double *d_kbar, bool *d_chi){
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
            if (TestBbox(xi+dx2, yi+dy2, zi+dz2) && !TestBit(d_dmg, NB*iind+b)) {
                int jind = iind + jadd[b];
                if(d_chi[jind]){
                    double psi = max(ZER, hrad - L0s[b])/hrad;
                    nsm += psi * d_W[jind] * NN * RM / Wt;
                    dsm += psi;
                }
            }
        }
        d_kbar[iind] = max(0.01, min(ONE, d_kbar[iind] + nsm / dsm));
    }
}

__global__ void updateK(double *d_k, double *d_kbar, int *d_NBCi, int *d_EBCi){
    int iind = blockIdx.x * blockDim.x + threadIdx.x;

    if(iind<NN){
        //Note: This condition may need to be modified depending on which boundary condition
        //regions should be included in the design domain.
        if(d_NBCi[iind]<0 && d_EBCi[iind]<0 && d_EBCi[NN + iind]<0 && d_EBCi[2*NN + iind]<0){
            d_k[iind] = alpha*d_k[iind] + (1 - alpha) * d_kbar[iind];
            d_kbar[iind] = ZER;
        }
    }
}


