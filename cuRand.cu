#include <iostream>
#include <iostream>
#include <string>
#include <random>
#include <ctime>


//cuda includes
#include "cuda_runtime.h"
#include <cuda.h>
#include <helper_cuda.h>
#include <cassert>
#include "curand.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);\
 return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
 printf("Error at %s:%d\n",__FILE__,__LINE__);\
 return EXIT_FAILURE;}} while(0)

//Generation of random numbers using cuda host API
int cudaHostRandom(size_t iterations)
{
 curandGenerator_t gen;
 unsigned int *devData, *hostData;
 /* Allocate n floats on host */
 hostData = (unsigned int *)calloc(iterations, sizeof(int));
 /* Allocate n floats on device */
 CUDA_CALL(cudaMalloc((void **)&devData, iterations*sizeof(int)));
 /* Create pseudorandom numbers from XORWOW generator */
 CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_XORWOW));
 /* Set seed */
CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,CURAND_ORDERING_PSEUDO_DEFAULT));
/* Generate n floats on device */
 CURAND_CALL(curandGenerate(gen, devData, iterations));
 /* Copy device memory to host */
 CUDA_CALL(cudaMemcpy(hostData, devData, iterations * sizeof(int),cudaMemcpyDeviceToHost));
 /* Show result */
 /*for(int i = 0; i < iterations; i++) {
 printf("%d", hostData[i]);
 }
 printf("\n");*/
 /* Clean */
 CURAND_CALL(curandDestroyGenerator(gen));
 CUDA_CALL(cudaFree(devData));
 free(hostData);
 return 0;
}

int curandHostNormal(int iterations)
{
curandGenerator_t gen;
 float *devData, *hostData;
 /* Allocate n floats on host */
 hostData = (float *)calloc(iterations, sizeof(float));
 /* Allocate n floats on device */
 CUDA_CALL(cudaMalloc((void **)&devData, iterations*sizeof(float)));
 /* Create pseudorandom numbers from XORWOW generator */
 CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
 /* Set seed */
CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,CURAND_ORDERING_PSEUDO_DEFAULT));
/* Generate n floats on device */
 CURAND_CALL(curandGenerateNormal(gen, devData, iterations,0,1));
 /* Copy device memory to host */
 CUDA_CALL(cudaMemcpy(hostData, devData, iterations * sizeof(float),cudaMemcpyDeviceToHost));
 /* Show result */
 /*for(int i = 0; i < iterations; i++) {
 printf("%1.4f ", hostData[i]);
 }
 printf("\n");*/
 /* Clean */
 CURAND_CALL(curandDestroyGenerator(gen));
 CUDA_CALL(cudaFree(devData));
 free(hostData);
 return 0;
}


int main()
{
  //Variables declaration
  size_t iterations = 6000000;
  
  cudaEvent_t start;
  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  checkCudaErrors(cudaEventRecord(start, NULL));


  float *normalized;
  normalized = (float *)calloc(iterations, sizeof(float));

  //Normal distribution
  std::random_device mch;
  //Seeding
  std::default_random_engine generator(mch()); 
  //Parameters are: Distribution mean, Standard deviation
  std::normal_distribution<double> distribution(0.0, 1.0); 
  for(int i=0; i<iterations; i++)
  {
    normalized[i] = distribution(generator);
    //printf("normal_distribution (0.0,1.0): %f\n", normalized[i]);
  }
free(normalized);
checkCudaErrors(cudaEventRecord(stop, NULL));
checkCudaErrors(cudaEventSynchronize(stop));

float msecTotal = 0.0f;
checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
double gigaFlops = (iterations * 1.0e-9f) / (msecTotal / 1000.0f);
printf("Basic normal processing time = %.3fms, \n Perf = %.3f Gflops\n", msecTotal, gigaFlops);


//Basic random distribution
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));
checkCudaErrors(cudaEventRecord(start, NULL));
  unsigned int *v1; 
  v1 = (unsigned int *)calloc(iterations, sizeof(int));
  //seeding
  srand((unsigned int)time(NULL)); 
  for (int i=0; i<iterations; i++)
  {
    //Just a basic random int 
    v1[i] = rand(); 
    //printf("Basic random number: %d\n", v1 [i]);
  }
free(v1);
checkCudaErrors(cudaEventRecord(stop, NULL));
checkCudaErrors(cudaEventSynchronize(stop));

msecTotal = 0.0f;
checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
gigaFlops = (iterations * 1.0e-9f) / (msecTotal / 1000.0f);
printf("Basic random processing time = %.3fms, \n Perf = %.3f Gflops\n", msecTotal, gigaFlops);


//cuda normal distribution on host
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));
checkCudaErrors(cudaEventRecord(start, NULL));

curandHostNormal(iterations);


checkCudaErrors(cudaEventRecord(stop, NULL));
checkCudaErrors(cudaEventSynchronize(stop));

msecTotal = 0.0f;
checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
gigaFlops = (iterations * 1.0e-9f) / (msecTotal / 1000.0f);
printf("Cuda host normal time = %.3fms, \n Perf = %.3f Gflops\n", msecTotal, gigaFlops);


//cuda basic random distribution on host
checkCudaErrors(cudaEventCreate(&start));
checkCudaErrors(cudaEventCreate(&stop));
checkCudaErrors(cudaEventRecord(start, NULL));

cudaHostRandom(iterations);

checkCudaErrors(cudaEventRecord(stop, NULL));
checkCudaErrors(cudaEventSynchronize(stop));

msecTotal = 0.0f;
checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
gigaFlops = (iterations * 1.0e-9f) / (msecTotal / 1000.0f);
printf("Cuda host random time = %.3fms, \n Perf = %.3f Gflops\n", msecTotal, gigaFlops);
 

return EXIT_SUCCESS;
}
