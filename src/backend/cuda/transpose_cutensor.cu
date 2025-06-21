#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cutensor.h>

#include <unordered_map>
#include <vector>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                             \
{ const auto err = x;                                               \
    if( err != CUTENSOR_STATUS_SUCCESS )                              \
    { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

extern "C" void launch_transpose_cutensor(
  const float* input, 
  float* output, 
  const int shape[4],   // host, 原来的shape
  const int perm[4],    // 采用的perm顺序，比如[0,1,2]->[1,0,2]，则perm={1,0,2,x}
  int rank              // rank
) {
    typedef float floatTypeA;   // input
    typedef float floatTypeC;   // output
    typedef float floatTypeCompute;

    cutensorDataType_t          const typeA       = CUTENSOR_R_32F;
    cutensorDataType_t          const typeC       = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t const descCompute = CUTENSOR_COMPUTE_DESC_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.0f;

    std::vector<int> modeC;
    std::vector<int> modeA;
    for(int i = 0; i < rank; ++i){
      modeA.push_back(i);
      modeC.push_back(perm[i]); 
    }

    int nmodeA = rank;
    int nmodeC = rank;

    std::unordered_map<int, int64_t> extent;
    extent[0] = shape[0];
    extent[1] = shape[1];
    extent[2] = shape[2];
    extent[3] = shape[3];

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);

    uint32_t const kAlignment = 128;          // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(input) % kAlignment == 0);
    assert(uintptr_t(output) % kAlignment == 0);

    /*************************
     * CUTENSOR
    *************************/

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                nullptr /* stride */,
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                nullptr /* stride */,
                                                typeC,
                                                kAlignment));

    /*******************************
     * Create Permutation Descriptor
     *******************************/

    cutensorOperationDescriptor_t  desc;
    HANDLE_ERROR(cutensorCreatePermutation(handle,
                                          &desc,
                                          descA,
                                          modeA.data(),
                                          CUTENSOR_OP_IDENTITY,
                                          descC,
                                          modeC.data(),
                                          descCompute));

    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                        CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                        (void*)&scalarType,
                                                        sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);

    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t  planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /**************************
     * Create Plan
     **************************/
  
    cutensorPlan_t  plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /* workspaceSizeLimit */));

    /**********************
     * Execute
     **********************/
    HANDLE_ERROR(cutensorPermute(handle,
      plan,
      &alpha, input, output, nullptr /* stream */));
}