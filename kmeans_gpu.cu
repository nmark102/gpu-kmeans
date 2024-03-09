#include "kmeans_gpu.h"

#define DEBUG 0

/**
 * Global constants
*/

/**
 * Global variables 
*/
// Z is hard-coded to 3 for 3-dimensional (image) vectors
size_t dimX = 0, dimY = 0, dimZ = 3, numMatrixElements = 0;

using namespace std;

__global__ void computeKMeans(int numClusters, float ** clusters, float * data,
                                size_t dimX, size_t dimY, size_t dimZ) {

    /**
     * Notes:
     * __shared__ qualifiers are for block-level variables
     * __device__ qualifiers are for  grid-level variables
    */
    extern __device__ float matrix[];

    /**
     * Some global variables:
     * "continue" flag
     * centroid allocation table
     * distance to centroid table
    */

    int continue = 1;

    while (continue) {
        
        // Clear centroid changed flag

        // Compute distance from points to centroids
        
        // Assign to nearest centroid

        // Set centroid changed flag

        __syncthreads();                // thread-level barrier

        // Compute centroids

        // Returns the number of threads whose predicate evaluates
        // to non-zero
        // continue = __syncthreads_count(CENTROID_CHANGED_FLAG_HERE);
    }
}

cudaError_t cudaInitAndCopy(void ** dst, void * src, size_t size) {
    cudaError_t cudaReturnCode;
    cudaReturnCode = cudaMalloc(dst, size);
    
    if (cudaReturnCode != 0)
        return cudaReturnCode;

    return cudaMemcpy(&dst, src, size, cudaMemcpyHostToDevice);
}

inline float * initMatrix() {
    string dimensions;

    // Parse dimensions    
    cin >> dimensions;
    dimX = atol(dimensions.c_str());

    cin >> dimensions;
    dimY = atol(dimensions.c_str());

    // Allocate memory and parse the next x*y*z floats as input
    size_t numMatrixElements = dimX * dimY * dimZ;
    float * matrix = new float[numMatrixElements];
        
    for (long i = 0; i < dimX * dimY; ++i) {
        string val;
        for (long j = 0; j < dimZ; ++j) {
            cin >> val;
            matrix[i * dimZ + j] = atof(val.c_str());
        }
    }

    return matrix;
}

inline float * initCentroids() {
    float * centroids = new float[numClusters * dimZ];

    srand(static_cast<unsigned> (time(NULL)));
    for (int i = 0; i < numClusters; ++i)
        centroids[i] = static_cast<float> (rand());

    return centroids;
}

inline char *  init

int main(int argc, char * argv[]) {
    
    int exitStatus = 0;

    // init data
    float * matrix = initMatrix();
    float * centroids = initCentroids();
    size_t sharedMemSize = sizeof(float) * (numMatrixElements + numClusters);

    // Move data to GPU
    float * matrixGpu    = nullptr;
    float * centroidsGpu = nullptr;

    cudaError_t cudaReturnCode;
    
    cudaReturnCode = cudaInitAndCopy((void **) &matrixGpu, (void *) matrix, 
                                        sizeof (float) * numMatrixElements);
    if (cudaReturnCode) {
        cerr << "Cannot copy matrix from host to CUDA device\n";
        exitStatus = 1;
        goto cleanup;
    }

    cudaReturnCode = cudaInitAndCopy((void **) &centroidsGpu, (void *) centroids, 
                                        sizeof (float) * numClusters);
    if (cudaReturnCode) {
        cerr << "Cannot copy clusters from host to CUDA device\n";
        exitStatus = 1;
        goto cleanup;
    }

    // Launching the kernel:
    
    // computeKMeans<<<gridSize, blockSize, sharedMemSize>>>(args);
    cudaDeviceSynchronize();    // wait to collect results

    // Transfer results back to CPU

cleanup:
    cudaFree(centroidsGpu);
    cudaFree(matrixGpu);

    delete centroids;
    delete matrix;

    cout << endl;
    return exitStatus;
}