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
int cudaHostPseudoRandom(size_t iterations)
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
 for(int i = 0; i < iterations; i++) {
 printf("%d", hostData[i]);
 }
 printf("\n");
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
 for(int i = 0; i < iterations; i++) {
 printf("%1.4f ", hostData[i]);
 }
 printf("\n");
 /* Clean */
 CURAND_CALL(curandDestroyGenerator(gen));
 CUDA_CALL(cudaFree(devData));
 free(hostData);
 return 0;
}


int main()
{
  //Variables declaration
  size_t iterations = 100;
  int v1[iterations];
  double normalized[iterations];
  //Normal distribution
  std::random_device mch;
  //Seeding
  std::default_random_engine generator(mch()); 
//Parameters are: Distribution mean, Standard deviation
  std::normal_distribution<double> distribution(0.0, 1.0); 
  for(int i=0; i<iterations; i++)
  {
    normalized[i] = distribution(generator);
    printf("normal_distribution (0.0,1.0): %f\n", normalized[i]);
  }

//Basic random distribution
  //seeding
  srand((unsigned int)time(NULL)); 
  for (int i=0; i<iterations; i++)
  {
    //Just a basic random int 
    v1[i] = rand(); 
    printf("Basic random number: %d\n", v1 [i]);
  }
 cudaHostPseudoRandom(iterations); 
 curandHostNormal(iterations);
 return EXIT_SUCCESS;
}
