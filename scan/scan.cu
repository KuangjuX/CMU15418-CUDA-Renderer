#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

const int MAXThreadsPerBlock = 1024;
const int MAXBlocksPerProcessor = 16;
const int MAXThreadsPerProcessor = MAXThreadsPerBlock * MAXBlocksPerProcessor;

extern float toBW(int bytes, float sec);

/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void prefix_sum_upsweep(int *device_data, int twod)
{
    int twod1 = twod * 2;
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * twod1;
    device_data[i + twod1 - 1] += device_data[i + twod - 1];
    __syncthreads();
}

__global__ void prefix_sum_downsweep(int *device_data, int twod)
{
    int twod1 = twod * 2;
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * twod1;
    int t = device_data[i + twod - 1];
    device_data[i + twod - 1] = device_data[i + twod1 - 1];
    device_data[i + twod1 - 1] += t;
    __syncthreads();
}

void check_upsweep(int *inarray, int *resultarray, int length)
{
    int mid = length / 2;
    int left_sum = 0, rigth_sum = 0;
    for (int i = 0; i < mid; i++)
    {
        left_sum += inarray[i];
    }
    for (int i = mid; i < length; i++)
    {
        rigth_sum += inarray[i];
    }
    assert(left_sum == resultarray[mid - 1]);
    assert(rigth_sum == resultarray[length - 1]);
}

void exclusive_scan(int *device_data, int length)
{
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */

    // Allocate blocks and threads.
    int blocks = 0;
    int threadsPerBlock = 0;

    // Execute upsweep phase.
    for (int twod = 1; twod < length; twod *= 2)
    {
        int twod1 = twod * 2;
        int threads = length / twod1;
        if (threads < 1024)
        {
            blocks = 1;
            threadsPerBlock = threads;
        }
        else
        {
            blocks = threads / 1024;
            threadsPerBlock = 1024;
        }
        // printf("blocks: %d, threadPreBlock: %d\n", blocks, threadsPerBlock);
        prefix_sum_upsweep<<<blocks, threadsPerBlock>>>(device_data, twod);
        // Wait all threads to execute.
        if (blocks > 1)
        {
            cudaThreadSynchronize();
        }
    }

    int data = 0;
    cudaMemcpy(device_data + length - 1, &data, sizeof(int), cudaMemcpyHostToDevice);

    for (int twod = length / 2; twod >= 1; twod /= 2)
    {
        int twod1 = twod * 2;
        int threads = length / twod1;
        if (threads < 1024)
        {
            blocks = 1;
            threadsPerBlock = threads;
        }
        else
        {
            blocks = threads / 1024;
            threadsPerBlock = 1024;
        }
        prefix_sum_downsweep<<<blocks, threadsPerBlock>>>(device_data, twod);
        cudaThreadSynchronize();
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int *inarray, int *end, int *resultarray)
{
    int *device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int length = end - inarray;
    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    if (rounded_length > length)
    {
        cudaMemset((void *)(device_data + length), 0, sizeof(int) * (rounded_length - length));
    }

    double startTime = CycleTimer::currentSeconds();

    // exclusive_scan(device_data, end - inarray);
    exclusive_scan(device_data, rounded_length);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);

    // check_upsweep(inarray, resultarray, rounded_length);
    // printf("Upsweep phase success");
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int *inarray, int *end, int *resultarray)
{

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void find_peaks_kernel(int *device_data, int *peak, int size)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id == 0 || id >= size - 1)
    {
        return;
    }
    if (device_data[id] > device_data[id - 1] && device_data[id] > device_data[id + 1])
    {
        peak[id] = 1;
    }
    __syncthreads();
}

__global__ void find_peaks_count(int *device_data, int *peak_counts, int *peak, int thread_size, int length)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int l = index * thread_size;
    int r = min(l + thread_size, length - 1);
    if (l == 0)
        l++;
    int count = 0;
    for (int i = l; i < r; i++)
    {
        if (device_data[i] > device_data[i - 1] && device_data[i] > device_data[i + 1])
        {
            count++;
            peak[i] = 1;
        }
    }
    peak_counts[index] = count;
    __syncthreads();
}

__global__ void count_peak(int *peak_counts, int length, int *count)
{
    *count = 0;
    for (int i = 0; i < length; i++)
    {
        *count += peak_counts[i];
    }
}

__global__ void generate_peaks(int *device_data, int *peak_index, int *peak, int thread_size, int length)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int l = index * thread_size;
    int r = min(l + thread_size, length);
    int pid = peak_index[index];

    for (int i = l; i < r; i++)
    {
        if (peak[i] == 1)
        {
            device_data[pid++] = i;
        }
    }

    __syncthreads();
}

int find_peaks(int *device_input, int length, int *device_output)
{
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */

    // int blocks = 0;
    // int threadsPerBlock = 0;
    // if (length <= MAXThreadsPerBlock)
    // {
    //     blocks = 1;
    //     threadsPerBlock = length;
    // }
    // else
    // {
    //     blocks = (length + MAXThreadsPerBlock - 1) / MAXThreadsPerBlock;
    //     threadsPerBlock = MAXThreadsPerBlock;
    // }

    // find_peaks_kernel<<<blocks, threadsPerBlock>>>(device_input, device_output, length);
    // cudaThreadSynchronize();

    // int *res_array = (int *)malloc(sizeof(int) * length);
    // int *output_array = (int *)malloc(sizeof(int) * length);
    // cudaMemcpy(res_array, device_output, sizeof(int) * length, cudaMemcpyDeviceToHost);

    // // exclusive_scan(device_output, length);

    // int count = 0;
    // for (int i = 0; i < length; i++)
    // {
    //     if (res_array[i] != 0)
    //     {
    //         output_array[count++] = i;
    //     }
    // }

    // cudaMemcpy(device_output, output_array, sizeof(int) * length, cudaMemcpyHostToDevice);

    // return count;

    int rounded_length = nextPow2(length);
    int totalThreads = 0;
    int thread_size = 0;
    if (length > MAXThreadsPerBlock)
    {
        totalThreads = MAXThreadsPerBlock;
        thread_size = rounded_length / MAXThreadsPerBlock;
    }
    else
    {
        totalThreads = length;
        thread_size = 1;
    }

    int *peak, *peak_index;
    cudaMalloc(&peak, sizeof(int) * length);
    cudaMalloc(&peak_index, sizeof(int) * nextPow2(totalThreads));

    printf("thread size: %d, totalThreads: %d\n", thread_size, totalThreads);
    find_peaks_count<<<1, totalThreads>>>(device_input, peak_index, peak, thread_size, length);
    cudaThreadSynchronize();

    int *device_count;
    cudaMalloc(&device_count, sizeof(int));
    count_peak<<<1, 1>>>(peak_index, totalThreads, device_count);

    int count = 0;
    cudaMemcpy(&count, device_count, sizeof(int), cudaMemcpyDeviceToHost);

    exclusive_scan(peak_index, nextPow2(totalThreads));
    generate_peaks<<<1, totalThreads>>>(device_output, peak_index, peak, thread_size, length);
    cudaFree(peak);
    cudaFree(peak_index);
    cudaFree(device_count);
    return count;
}

/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length)
{
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("  Maximum number of blocks per multiprocessor: %d\n", deviceProps.maxBlocksPerMultiProcessor);
        printf("  Maximum number of threads per multiprocessor: %d\n", deviceProps.maxThreadsPerMultiProcessor);
        printf("  Maximum number of blocks: %d\n", deviceProps.maxBlocksPerMultiProcessor * deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
