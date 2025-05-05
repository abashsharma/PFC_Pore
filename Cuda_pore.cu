#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <bits/stdc++.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void Pore(double *g, int N, int real_size, double r0, double r1, double xi, double yi){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x,y;
    double rad;
    double f;
    if (i<real_size){
        x = (i % (N));
        y= int(i / (N));
        if(x < N) {
            rad = powf((x - xi), 2) + powf((y - yi), 2);
            rad = powf(rad, 0.5);
            if (rad <= r0) {
                 f = 1.0;
            } else if (rad > r0 && rad <= r1) {
                f = 2.0 * powf(((rad - r0) / (r1 - r0)), 3) - 3.0 * powf(((rad - r0) / (r1 - r0)), 2) + 1.0;
            } else {
                f = 0.0;
            }

            g[i] = 1.0 - f;

        }
        else{
            g[i] = 0.0f;
        }
    }

    __syncthreads();
}

__global__ void PFC(double *data,double *data2, double *data3,double *data4, double *gpsi, double *g, int N, int real_size, double r, double chi, double phi0){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x,y;
    double rad;
    double f;
    if (i<real_size){
        x = (i % (N));
        if(x < N) {
           f=1.0-g[i];

            data3[i] = ((r+1.0)*data[i] + data2[i] + data[i]*data[i]*data[i] + data4[i]*0.5)*g[i]  + chi*(data[i]-phi0)*f;
            gpsi[i]=g[i]*data[i];

        }
        else{
            data3[i] = 0.0f;
            gpsi[i]=0.0f;
        }
    }

    __syncthreads();
}


__global__ void Free_en(double *data,double *data2,double *data4, int real_size, int N, double *g, double chi, double phi0, double psi0, double r){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x,y;
    float f;
    if (i<real_size) {
        x = (i % (N));
        y = int(i / N);
        if (x < N) {
            f = 1 - g[i];

            data2[i] = (0.5 * (1.0 + r) * data[i] * data[i] + data[i] * data2[i] + 0.5 * data[i] * data4[i] +
                       0.25 * powf(data[i], 4)) * g[i] + chi * 0.5 * (data[i] - phi0) * (data[i] - phi0) * f;
            //if (i==100){printf("%f %f %f %f %f\n",f, g[i], data[i], data2[i], data4[i]);}
        } else {
            data2[i] = 0.0f;
        }
    }
    __syncthreads();
}


__global__ void Add(double *data2, double *data3,double *data4, int N, int real_size){


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x;
    double g_local;
    if (i<real_size){
        x = (i % (N));
        if(x < N) {

            data3[i] = data3[i] + data2[i] + data4[i]*0.5;
        }
        else{
            data3[i] = 0.0f;
        }
    }

    __syncthreads();
}


__global__ void Integration(double *data,double *data2, int N, int real_size, double dt){


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x;
    if (i<real_size){
        x = (i % (N));
        if(x < N){

            data[i]=data[i]+dt*data2[i];

        }
        else{
            data[i] = 0.0f;
        }
    }

    __syncthreads();
}


__global__ void Laplace(double *data, double *data2, int N, int real_size, int n,
                        int *d_1, int *d_2, int *d_3, int *d_4, int *d_5, int *d_6, int *d_7, int *d_8, double dn) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int x;
    double a=dn*dn;

    if (i<real_size){
        x = (i % (N));
        if(x < N){
            int neb1=d_1[i];
            int neb2=d_2[i];
            int neb3=d_3[i];
            int neb4=d_4[i];
            int neb5=d_5[i];
            int neb6=d_6[i];
            int neb7=d_7[i];
            int neb8=d_8[i];

            data2[i] = ((data[neb1] + data[neb3] + data[neb5] + data[neb7])*0.5 + (data[neb2]+data[neb4] + data[neb6] + data[neb8])*0.25 - 3*data[i])/a;

        }
        else{
            data2[i] = 0.0f;
        }
    }

    __syncthreads();
}


__global__ void Link ( int *d_1, int *d_2, int *d_3, int *d_4, int *d_5, int *d_6, int *d_7, int *d_8, int N, int real_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<real_size) {
        int x = (i % (N));
        int y = int(i / (N));
        if (x < N) {

            d_1[i] = i + 1;
            d_2[i] = i + 1 + (N);
            d_3[i] = i + (N);
            d_4[i] = i - 1 + (N);
            d_5[i] = i - 1;
            d_6[i] = i - 1 - (N);
            d_7[i] = i - (N);
            d_8[i] = i + 1 - (N);

            if (x == 0) {
                d_6[i] = i - 1;
                d_5[i] = i + N - 1;
                d_4[i] = i + 2 * N - 1;
            }

            if (x == (N - 1)) {
                d_2[i] = i + 1;
                d_1[i] = i + 1 - N;
                d_8[i] = i + 1 - 2 * N;
            }

            if (y == 0) {
                d_6[i] = i - 1 + real_size - (N);
                d_7[i] = i + real_size - (N);
                d_8[i] = i + 1 + real_size - (N);
            }

            if (y == N - 1) {
                d_2[i] = i + 1 + (N) - real_size;
                d_3[i] = i - real_size + (N);
                d_4[i] = i - real_size + (N) - 1;
            }


            if (x == 0 && y == 0) {
                d_6[i] = real_size - 1;
            }

            if (x == N - 1 && y == 0) {
                d_8[i] = real_size - (N);
            }


            if (x == 0 && y == N - 1) {
                d_4[i] = N - 1;
            }

            if (x == N - 1 && y == N - 1) {
                d_2[i] = 0;
            }
        }
    }
    __syncthreads();
}



double ran();

void load_parameters(int &N, double &psi0, double &r, double &dn, double &dt, double &chi, double &phi0, double &tnuc, int &Rmax, double &nuc, int **seeds, int &num_seeds, double &r0, double &r1,
                     double &xi, double &yi );


void seed(double *h_data, double r0, double r1, double xi, double yi,  double psi0,double phi0, int N, int real_size);
void cont(double *h_data, int N, int real_size);


void grow(double *phi, double *h_data,double *FE_data, double *d_data,double *d_data2, double *d_data3, double *d_data4,double *g, double *gpsi,
          int &tau, int N, int n, int real_size, int mem_size,
          double identity, double psi0, double r, double dt, double chi, double phi0, double r0,double r1, double dn,
          int numBlocks, int threadsPerBlock, double xi, double yi, int *d_1, int *d_2, int *d_3, int *d_4, int *d_5, int *d_6, int *d_7, int *d_8,
          int *nbr, int *size, int *list, int *head, int *clst, double *xcom, double *ycom, double A, double nuc, int &atomconst);

void Free_energy(double *FE_data,int N, int real_size, double psi0);

void particle_count(int *nbr, double *phi, int N, int tau, double psi0, double r,
                    double dn, double dt, int real_size, int *size, int *list, int *head, int *clst, double *xcom, double *ycom, double identity, double A, double nuc, int &atomconst, double name);

void a_linkedlist(int N, int maxatoms, double cutoff, int &atom, double *xcom, double *ycom, double *phi, int *size, int *list, int *head, int *clst, int &atomconst);

void outputcom(double psi0, int N, double dn, double dt, double r, double identity, int tau, double *xcom, double *ycom, int atom, int atomconst, double name);

void image_output(double *h_data, double *phi, double name, int real_size, int N, int A);

void output_data(double *h_data, int N, int tau, double xi, double yi, double r0, double r1, double psi0, double phi0);


int main (int argc, char **argv) {


    //give the random number generator a seed based on the current time
    int seedtime = time(NULL);
    //or use the same seed to repeatably generate an initial system state
    //seedtime = 1523551688;
    srand(seedtime);
    //cout <<seedtime <<endl;


    //determine which GPU to use based on available free memory

    int prime, start;
    int count = 0;

    size_t total_mem, free_mem;
    int gpu1mem=0, gpu0mem=0;

    //if free memory < 2GB, wait and retry
    while (gpu1mem<2 && gpu0mem<2){
        cudaSetDevice(0);
        cudaMemGetInfo(&free_mem, &total_mem);
        gpu0mem=free_mem/pow(10,9);

        cudaSetDevice(1);
        cudaMemGetInfo(&free_mem, &total_mem);
        gpu1mem=free_mem/pow(10,9);

        //this is meant to take up some time between retries if GPU memory is full
        if(gpu1mem<2 && gpu0mem<2){
            start = time(0);
            prime = (int)abs(ran()*100000000000);
            for(int i=2; i<prime; i++){
                if(prime%i==0){count++;}
            }
            //cout <<"prime calc time " << (time(0) - start) <<endl <<endl;
        }
    }

    //set the GPU we want to use
    int setgpu;
    if(gpu1mem<gpu0mem){cudaSetDevice(0); setgpu=0;}
    else{setgpu=1;}
    cudaMemGetInfo(&free_mem, &total_mem);

    //defining some variables
    double dt;
    double psi0, r, dn, chi,phi0, tnuc, nuc, identity, r0, r1,xi,yi;
    int N, Rmax, n, tau, num_seeds;

    //unique identifier for each new trial
    identity = round(10000*abs(ran()));

    int **seeds;
    seeds = new int *[1000];



    //load parameters from text file
    load_parameters(N, psi0, r, dn, dt, chi, phi0, tnuc, Rmax, nuc, seeds, num_seeds, r0, r1,xi,yi);

    Rmax = 0;
    num_seeds = N;
    int atomconst = 0;

    n = N * N;

    int maxparticle = 0;
    double A;
    A = -4.0/5.0*( -nuc + 1.0/3.0*sqrt( abs(-15.0*r-36.0*nuc*nuc) ) );

    double *phi = new double[n];

    //declaring linked list arrays
    int *size = new int[n];
    int *list = new int[n];
    int *head = new int[n];
    int *clst = new int[n];
    double *xcom = new double[n];
    double *ycom = new double[n];
    int *nbr = new int[n];


    int *seedsize = NULL;
    seedsize = new int[num_seeds];

    int real_size=n;
    int mem_size = sizeof(double)*n;
    int mem_size_nbr = sizeof(int)*n;

    double *h_data = (double*)malloc(mem_size);
    double *FE_data = (double*)malloc(mem_size);
    double *d_data;
    gpuErrchk(cudaMalloc((void**)&d_data, mem_size));

    double *d_data2;
    gpuErrchk(cudaMalloc((void**)&d_data2, mem_size));

    double *d_data3;
    gpuErrchk(cudaMalloc((void**)&d_data3, mem_size));

    double *d_data4;
    gpuErrchk(cudaMalloc((void**)&d_data4, mem_size));

    double *g;
    gpuErrchk(cudaMalloc((void**)&g, mem_size));
    double *gpsi;
    gpuErrchk(cudaMalloc((void**)&gpsi, mem_size));

    int *d_1;
    gpuErrchk(cudaMalloc((void**)&d_1, mem_size_nbr));
    int *d_2;
    gpuErrchk(cudaMalloc((void**)&d_2,mem_size_nbr));
    int *d_3;
    gpuErrchk(cudaMalloc((void**)&d_3, mem_size_nbr));
    int *d_4;
    gpuErrchk(cudaMalloc((void**)&d_4, mem_size_nbr));
    int *d_5;
    gpuErrchk(cudaMalloc((void**)&d_5, mem_size_nbr));
    int *d_6;
    gpuErrchk(cudaMalloc((void**)&d_6, mem_size_nbr));
    int *d_7;
    gpuErrchk(cudaMalloc((void**)&d_7, mem_size_nbr));
    int *d_8;
    gpuErrchk(cudaMalloc((void**)&d_8, mem_size_nbr));



    //CUDA kernel variables (this may not be optimized, see documentation)
    int maxThreads=(N>1024)?1024:N;
    int threadsPerBlock = maxThreads;
    //int numBlocks = n/maxThreads;
    int numBlocks = real_size/threadsPerBlock;
    //cout <<"t per block "<<threadsPerBlock <<" blocks " <<numBlocks <<endl;

    cudaMemGetInfo(&free_mem, &total_mem);
    //cout <<"free "<<free_mem/pow(10,9) <<" total " <<total_mem/pow(10,9)  <<" used " <<(total_mem - free_mem)/pow(10,9) <<" GB" <<endl <<endl;

    double Ravg=0.0;

//////////////////////////////////////////////////////////////////////////////////////////////////////

//create particles in the system using the previously generated sizes and positions
    seed(h_data, r0,r1,xi,yi, psi0,phi0, N, real_size);
    //cont(h_data, N, real_size);  //to continue from the middle

//start the time evolution of the system
    grow(phi, h_data,FE_data, d_data, d_data2, d_data3,d_data4,g,gpsi, tau, N, n, real_size, mem_size,identity, psi0, r, dt, chi, phi0,r0,r1, dn, numBlocks, threadsPerBlock, xi, yi,
         d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, nbr, size, list, head, clst, xcom, ycom, A, nuc, atomconst);


    //free up the memory on the GPUs
    cudaFree(d_1);
    cudaFree(d_2);
    cudaFree(d_3);
    cudaFree(d_4);
    cudaFree(d_5);
    cudaFree(d_6);
    cudaFree(d_7);
    cudaFree(d_8);
    cudaFree(g);
    cudaFree(gpsi);
    cudaFree(d_data);
    cudaFree(d_data2);
    cudaFree(d_data3);
    cudaFree(d_data4);


    free(h_data);
    free(FE_data);

    return 0;
}


void seed(double *h_data, double r0, double r1, double xi, double yi, double psi0,double phi0, int N, int real_size){

    double r;
    for (int i = 0; i < real_size; i++) {
        double b = ran();
    }


    for (int i = 0; i < real_size; i++) {
        if (i < real_size) {
            int x = (i % (N));
            int y = int(i / (N));
            if (x < N) {
                r = powf((x - xi), 2) + powf((y - yi), 2);
                r = powf(r, 0.5);

                if (r <= r0) {
                    h_data[i] = phi0;
                }
                else if (r > r0 && r<=r1 ) {
                    h_data[i] = psi0/2.0;
                }
                else {
                    //h_data[i] = psi0+ 0.1 * ran();
                    h_data[i] = psi0;
                }
            }
            else {
                h_data[i] = 0.0f;
            }
        }
    }
}

void cont(double *h_data, int N, int real_size){
    int i=0;

    char temp1[128],temp2[128],temp3[128];

    ifstream fin;
    fin.open("data.csv");

    while(fin>>temp1>>temp2>>temp3){
        h_data[i]=atof(temp3);
        i++;
    }

    if(i!=real_size){throw 0;}

    fin.close();


}



void grow(double *phi, double *h_data, double *FE_data, double *d_data,double *d_data2, double *d_data3, double *d_data4,double *g, double *gpsi,
          int &tau, int N, int n, int real_size, int mem_size,
          double identity, double psi0, double r, double dt, double chi, double phi0, double r0,double r1, double dn,
          int numBlocks, int threadsPerBlock, double xi, double yi, int *d_1, int *d_2, int *d_3, int *d_4, int *d_5, int *d_6, int *d_7, int *d_8,
          int *nbr, int *size, int *list, int *head, int *clst, double *xcom, double *ycom, double A, double nuc, int &atomconst){



    double psi_avg, rad;
    gpuErrchk(cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    Link<< < numBlocks, threadsPerBlock >> >(d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, N, real_size);
    cudaDeviceSynchronize();


    Pore << < numBlocks, threadsPerBlock >> >
                         (g, N, real_size, r0,r1,xi, yi);
    cudaDeviceSynchronize();


    for (tau = 0; tau < 100000001; tau++) {


        //laplace of 1st into second : delsq*psi
        Laplace << < numBlocks, threadsPerBlock >> >
                                (d_data, d_data2, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
        cudaDeviceSynchronize();

        //laplace of laplace of psi:del4*psi
        Laplace << < numBlocks, threadsPerBlock >> >
                                (d_data2, d_data4, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
        cudaDeviceSynchronize();

        //Add all the normal PFC terms
        PFC << < numBlocks, threadsPerBlock >> >
                            (d_data, d_data2, d_data3, d_data4, gpsi, g, N, real_size, r, chi, phi0);
        cudaDeviceSynchronize();

        Laplace << < numBlocks, threadsPerBlock >> >
                                (gpsi, d_data2, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
        cudaDeviceSynchronize();

        Laplace << < numBlocks, threadsPerBlock >> >
                                (d_data2, d_data4, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
        cudaDeviceSynchronize();

        Add << < numBlocks, threadsPerBlock >> >
                            (d_data2, d_data3, d_data4, N, real_size);
        cudaDeviceSynchronize();

        Laplace << < numBlocks, threadsPerBlock >> >
                                (d_data3, d_data2, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
        cudaDeviceSynchronize();

        //Laplacian and Integration
        Integration << < numBlocks, threadsPerBlock >> >
                                    (d_data, d_data2, N, real_size, dt);
        cudaDeviceSynchronize();

        if (tau % 500000 == 0) {

            //Calculation of Free Energy
            Laplace << < numBlocks, threadsPerBlock >> >
                                    (d_data, d_data2, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
            cudaDeviceSynchronize();
            //laplace of laplace of psi:del4*psi
            Laplace << < numBlocks, threadsPerBlock >> >
                                    (d_data2, d_data4, N, real_size, n, d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, dn);
            cudaDeviceSynchronize();
            Free_en << < numBlocks, threadsPerBlock >> >
                                    (d_data, d_data2, d_data4, real_size, N, g, chi, phi0, psi0, r);
            cudaDeviceSynchronize();

            gpuErrchk(cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost));
            gpuErrchk(cudaMemcpy(FE_data, d_data2, mem_size, cudaMemcpyDeviceToHost));

            output_data (h_data, N, tau, xi, yi, r0, r1, psi0, phi0);

            particle_count(nbr, h_data, N, tau, psi0, r, dn, dt,
                           real_size, size, list, head, clst, xcom, ycom, identity, A, nuc, atomconst, phi0);


            if (tau%10000000==0) {
                Free_energy(FE_data, N, real_size, psi0);
                image_output(h_data,phi, psi0,real_size, N,tau);}

            gpuErrchk(cudaMemcpy(d_data, h_data, mem_size, cudaMemcpyHostToDevice));

        }
    }
    gpuErrchk(cudaMemcpy(h_data, d_data, mem_size, cudaMemcpyDeviceToHost));

}


void load_parameters(int &N, double &psi0, double &r, double &dn, double &dt, double &chi, double &phi0, double &tnuc, int &Rmax, double &nuc, int **seeds, int &num_seeds,
                     double &r0, double &r1, double &xi, double &yi){

    N =1024;
    psi0 = -0.290;
    r = -0.2;
    dn = 0.8;
    dt = 0.01;
    chi =10;
    phi0 = -0.3;
    //cin >> phi0;
    r0 = 50;
    r1 = 60;
    xi=(N/2)-1;
    yi=(N/2)-1;
    nuc = -0.2667;

}


double ran(){    //random in range [-1,1]
    double u= double (rand())/(RAND_MAX);
    return 2*u-1;
}

double uniran(){    //random in range [0,1]
    double u = double (rand())/(RAND_MAX);
    return u;
}


void particle_count(int *nbr, double *phi, int N, int tau, double psi0, double r, double dn, double dt,
                    int real_size, int *size, int *list, int *head, int *clst, double *xcom, double *ycom, double identity, double A, double nuc, int &atomconst, double name){

    int u;
    int n;


    double peak_cutoff = nuc+abs(A/5.0);
    //cout <<"peak_cutoff " <<peak_cutoff <<endl;

    int atom = 0;
    n = N * N;
    int maxatoms = n;

    for(int i=0; i<maxatoms; i++){
        xcom[i] = 0; ycom[i] = 0;
    }

    a_linkedlist(N, maxatoms, peak_cutoff, atom, xcom, ycom, phi, size, list, head, clst, atomconst);

    outputcom(psi0, N, dn, dt, r, identity, tau, xcom, ycom, atom, atomconst, name);


}

void a_linkedlist(int N, int maxatoms, double cutoff, int &atom, double *xcom, double *ycom, double *phi, int *size, int *list, int *head, int *clst, int &atomconst){

    int n=N*N;

    int nbr[4];

    int n0const =  N-1;
    int n1const = -N+1;
    int n2const = -N+n;
    int n3const =  N-n;

    int Nclst = 0;

    for(int i=0; i<maxatoms; i++){
        size[i]=0;
        head[i]=-1;
        clst[i]=-1;
        list[i]=-1;
    }

    //cout <<"starting xy loop" <<endl;
    int toggle=0;
    int i, ii, jj, I, J;
    int itr=0;

    for (int y=0; y<N; y++){
        for (int x=0; x<N; x++){


            i = x + y*N;

            if (phi[i]>cutoff){
                ii = clst[i];
                nbr[0]=i-1;
                nbr[1]=i+1;
                nbr[2]=i-N;
                nbr[3]=i+N;

                if (x==0){
                    nbr[0] = i + n0const;
                }
                if (x==N-1){
                    nbr[1] = i + n1const;
                }
                if (y==0){
                    nbr[2] = i + n2const;
                }
                if (y==N-1){
                    nbr[3] = i + n3const;
                }



                for (int j=0; j<4; j++){


                    if(phi[nbr[j]]>cutoff){
                        ii = clst[i];
                        jj = clst[nbr[j]];


                        if(ii == -1 and jj == -1){	//if neither i nor j are part of a cluster
                            head[itr]=i;			//create new cluster for i and j
                            clst[i]=itr;	 		//add i and j to the cluster
                            clst[nbr[j]]=itr;
                            list[i]=nbr[j];			//add j to the list
                            size[itr]=2;			//update the size of the cluster
                            itr++;
                            Nclst++;
                        }

                        if(ii!=jj){

                            if(ii==-1 and jj>=0){			//if j is part of a cluster but i is not
                                //if(toggle==1){cout <<"a" <<endl;}
                                clst[i]=clst[nbr[j]];		//add i to jj
                                list[i]=head[jj];	//add i on top of jj
                                head[jj]=i;		//make i the new head of jj
                                size[jj]++;
                            }

                            if(jj==-1 and ii>=0){			//if is is part of a clst but j is not
                                //if(toggle==1){cout <<"b" <<endl;}
                                clst[nbr[j]]=clst[i];
                                list[nbr[j]]=head[ii];
                                head[ii]=nbr[j];
                                size[ii]++;
                            }



                            if(ii>=0 and jj>=0){				//if both i and j are part of clusters already


                                if(ii<jj){						//add j to i

                                    clst[head[jj]]=ii;					//add the head of jj to ii
                                    int ctr = 1;
                                    J=head[jj];
                                    while(list[J]>=0){
                                        J=list[J];
                                        if(toggle==1){cout <<J <<endl;}
                                        clst[J]=ii;				//add the rest of jj to ii
                                        ctr++;
                                    }


                                    list[J]=head[ii];			//add the last element of jj on top of ii
                                    head[ii]=head[jj];			//make the head of jj the new head of ii
                                    size[ii]+=size[jj];			//update the size of jj
                                    size[jj]=0;

                                    ctr=1;
                                    I=head[ii];
                                    while(list[I]>=0){
                                        I=list[I];
                                        ctr++;
                                    }

                                    Nclst--;					//subtract a cluster from the count
                                }

                                if(jj<ii){						//add i to j

                                    I=head[ii];
                                    clst[head[ii]]=jj;
                                    int ctr =1;
                                    while(list[I]>=0){
                                        I=list[I];

                                        clst[I]=jj;
                                        ctr++;
                                    }

                                    list[I]=head[jj];
                                    head[jj]=head[ii];
                                    size[jj]+=size[ii];
                                    size[ii]=0;

                                    ctr=1;
                                    J=head[jj];
                                    while(list[J]>=0){
                                        J=list[J];
                                        ctr++;
                                    }


                                    Nclst--;

                                }
                            }
                        }
                    }
                }
            }
        }
    }

    //cout <<"Nclst " <<Nclst <<" itr " <<itr <<endl;
    int constitr=itr;
    int max=itr;
    for(int i=0; i<max; i++){
        while(size[max-1]<1){max--;}
        if(size[i]==0){
            size[i]=size[max-1];
            size[max-1]=0;
            head[i]=head[max-1];
            head[max-1]=-1;
            I=head[max-1];
            while(I>=0){
                clst[I]=i;
                I=list[I];
            }
            max--;
        }
        for(int j=0; j<constitr; j++){
            if(size[j]>0){max=j+1;}
        }
        //cout <<"max " <<max <<" " <<size[max-1]<<endl;
    }

    atom = max;
    //cout <<"# atoms "<<atom <<endl;

    for(int i=0; i<atom; i++){
        if(size[i]==0){
            cout <<i <<" atom size = 0" <<endl;
            //exit (EXIT_FAILURE);
        }
    }

    //////////////////////////////////////////////////////////////

    //find center of mass

    int x, y, peak;

    for(int i=0; i<Nclst; i++){

        I=head[i];
        //asize[i]++;
        //mass[i]+=phi[I];
        x = I % N;
        y = (I / N) % N;
        xcom[i] = x;
        ycom[i] = y;
        peak = I;
        while(list[I]>=0){

            I=list[I];
            if(phi[I]>peak){peak=I;}

        }
        xcom[i] = peak % N;
        ycom[i] = ( peak / N ) % N;
    }

    double ycold, xcold;

    for (int i=0; i<atom; i++){
        ycold=ycom[i];
        xcold=xcom[i];
        if(xcold>=N){xcom[i]-=N;}
        if(xcold<0){xcom[i]+=N;}
        if(ycold>=N){ycom[i]-=N;}
        if(ycold<0){ycom[i]+=N;}

    }

    //if(atomconst == 0){atomconst = 1.1*atom;}
    atomconst = 20000;
}



void image_output(double *h_data,double *phi, double name, int real_size, int N, int A){

    int n=N*N;
    for (int i=0; i<n; i++){
        phi[i]=h_data[i];
    }

    ofstream fout;
    string ofilename;

    ostringstream outstr;
    outstr.precision(4);
    outstr <<"psi_"<<name<<"_"<<A<<"_data_n"<<".pgm";
    ofilename=outstr.str();
    fout.open(ofilename.c_str(), std::ofstream::out);

    //PGM format
    //black is 0 and 255 is white

    fout <<"P2" <<endl <<N <<" " <<N <<endl <<255 <<endl;

    double datamin=1; double datamax=-1;
    for(int i=0; i<n; i++){
        if(datamin>phi[i]){datamin=phi[i];}
        if(datamax<phi[i]){datamax=phi[i];}
    }

    for(int i=0; i<n; i++){
        phi[i]-=datamax;
    }
    datamin-=datamax;
    datamin=255/datamin;
    for(int i=0; i<n; i++){
        phi[i]=255-phi[i]*datamin;
    }

    for(int i=0; i<n; i++){
        fout <<round(phi[i]) <<", ";
        if(i%N==0){fout <<endl;}
    }

    fout.close();
}

void outputcom(double psi0, int N, double dn, double dt, double r, double identity, int tau, double *xcom, double *ycom, int atom, int atomconst, double name){



    ofstream fout;
    ostringstream outstr;
    outstr.precision(4);
    outstr <<"Psi_"<<psi0<<"_atom_coords.xyz";
    string ofilename=outstr.str();
    fout.open(ofilename.c_str(), std::ios::app);

    fout.precision(5);


    int acount = 0;
    if(fout.is_open()){
        fout <<atomconst <<endl;
        fout <<"test" <<endl;
        for (int i=0; i<atom; i++){
            fout <<1 <<" "<<xcom[i] <<" " <<ycom[i] <<" " <<0.0 <<endl;
            acount++;
        }

        if(atom<atomconst){
            for(int i=0; i<atomconst-atom; i++){
                fout <<1 <<" " <<0 <<" " <<0 <<" " <<0.0 <<endl;
                acount++;
            }
        }

        //fout <<endl;

        fout.close();
    }

    else{cerr <<"file not opened" <<endl; throw 0;}

    ostringstream outstr3;
    outstr3 <<"psi_"<<psi0<< "_no_atoms.csv";
    ofilename = outstr3.str();
    fout.open(ofilename.c_str(), std::ios::app);
    fout.precision(5);

    if (fout.is_open()) {
        fout << tau << " " << atom << endl;
        fout.close();
    }
    else {cerr << "file not opened" << endl; throw 0;}


}

void output_data(double *h_data, int N, int tau, double xi, double yi, double r0, double r1, double psi0, double phi0){


    int n=N*N;
    int count=0;
    double psi_avg=0;
    double psi_total=0;
    double rad=0;


    ofstream fout;
    ostringstream outstr;
    outstr.precision(4);
    outstr <<"psi_"<<psi0<< "_Total_Psiavg.csv";
    string ofilename=outstr.str();
    fout.open(ofilename.c_str(), std::ios::app);

    fout.precision(5);


    if(fout.is_open()) {


        for (int i = 0; i < n; i++) {

            int x = (i % (N));
            int y = int(i / (N));
            if (x < N) {

                rad = powf((x - xi), 2) + powf((y - yi), 2);
                rad = powf(rad, 0.5);
                psi_total += h_data[i];
                if (rad > r1) {
                    psi_avg += h_data[i];
                    count += 1;
                }
            }
        }
        fout << tau << " " << psi_total / n << endl;
        fout.close();
    }
    else{cerr <<"file not opened" <<endl; throw 0;}


    ostringstream outstr2;
    outstr2 <<"psi_"<<psi0<< "_Psiavg.csv";
    ofilename = outstr2.str();
    fout.open(ofilename.c_str(), std::ios::app);
    fout.precision(5);

    if(fout.is_open()) {
        fout << tau << " " << psi_avg / count << endl;
        fout.close();
    }
    else{cerr <<"file not opened" <<endl; throw 0;}


    if (tau%10000000==0) {
        ostringstream outstr3;
        outstr3 <<"psi_"<<psi0<<"data"<<tau<<".csv";
        ofilename = outstr3.str();
        fout.open(ofilename.c_str(), std::ofstream::out);

        for (int i = 0; i < n; i++) {
            int x = (i % (N));
            int y= int(i / (N));
            if (x < N) {
                fout << x << " "<<y<<" "<< h_data[i]<<endl;

            }
        }
        fout.close();

        ostringstream outstr4;
        outstr4 <<"psi_"<<psi0<<"_n_profile"<<tau<<".csv";
        ofilename = outstr4.str();
        fout.open(ofilename.c_str(), std::ofstream::out);

        for (int i = 0; i < n; i++) {
            int x = (i % (N));
            int y= int(i / (N));
            if (x < N) {
                if (y==(N/2)-1) {
                    fout << x << " " << h_data[i] << endl;
                }
            }
        }
        fout.close();

    }


}

void Free_energy(double *FE_data, int N, int real_size, double psi0){


    ofstream fout;
    ostringstream outstr;
    outstr.precision(4);
    outstr <<"psi_"<<psi0<< "_Free_energy.csv";
    string ofilename=outstr.str();
    fout.open(ofilename.c_str(), std::ofstream::out);

    fout.precision(5);


    if(fout.is_open()) {


        for (int i = 0; i < real_size; i++) {
            int x = (i % (N));
            int y= int(i / (N));
            if (y==(N/2)-1) {fout << x << ", " << FE_data[i] << endl;}
        }
    }

    else{cerr <<"file not opened" <<endl; throw 0;}
    fout.close();
}
