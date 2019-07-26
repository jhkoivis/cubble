#pragma once

#include "Util.h"
#include "Vec.h"
#include <assert.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace cubble
{
extern __constant__ __device__ double dTotalArea;
extern __constant__ __device__ double dTotalFreeArea;
extern __constant__ __device__ double dTotalFreeAreaPerRadius;
extern __constant__ __device__ double dTotalVolume;
extern __device__ bool dErrorEncountered;
extern __device__ int dNumPairs;
extern __device__ double dVolumeMultiplier;
extern __device__ double dInvRho;

template <typename... Arguments>
void cudaLaunch(const char *kernelNameStr, const char *file, int line,
                void (*f)(Arguments...), KernelSize kernelSize,
                uint32_t sharedMemBytes, cudaStream_t stream, Arguments... args)
{
#ifndef NDEBUG
  assertMemBelowLimit(kernelNameStr, file, line, sharedMemBytes);
  assertBlockSizeBelowLimit(kernelNameStr, file, line, kernelSize.block);
  assertGridSizeBelowLimit(kernelNameStr, file, line, kernelSize.grid);
#endif

  f<<<kernelSize.grid, kernelSize.block, sharedMemBytes, stream>>>(args...);

#ifndef NDEBUG
  CUDA_ASSERT(cudaDeviceSynchronize());
  CUDA_ASSERT(cudaPeekAtLastError());

  bool errorEncountered = false;
  void *dee             = nullptr;
  CUDA_ASSERT(cudaGetSymbolAddress((void **)&dee, dErrorEncountered));
  if (dee != nullptr)
    CUDA_ASSERT(cudaMemcpy(static_cast<void *>(&errorEncountered), dee,
                           sizeof(bool), cudaMemcpyDeviceToHost));
  else
    throw std::runtime_error(
      "Couldn't get symbol address for dErrorEncountered variable!");

  if (errorEncountered)
  {
    std::stringstream ss;
    ss << "Error encountered during kernel execution."
       << "\nError location: '" << kernelNameStr << "' @" << file << ":" << line
       << "."
       << "\nSee earlier messages for possible details.";

    throw std::runtime_error(ss.str());
  }
#endif
}

__device__ void logError(bool condition, const char *statement,
                         const char *errMsg);

__device__ int getGlobalTid();

__device__ void resetDoubleArrayToValue(double value, int idx, double *array);

template <typename... Args>
__device__ void resetDoubleArrayToValue(double value, int idx, double *array,
                                        Args... args)
{
  resetDoubleArrayToValue(value, idx, array);
  resetDoubleArrayToValue(value, idx, args...);
}

template <typename... Args>
__global__ void resetKernel(double value, int numValues, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
    resetDoubleArrayToValue(value, tid, args...);
}

template <typename T>
__device__ void copyValue(int fromIndex, int toIndex, T *fromArray, T *toArray)
{
  toArray[toIndex] = fromArray[fromIndex];
}

template <typename T, typename... Args>
__device__ void copyValue(int fromIndex, int toIndex, T *fromArray, T *toArray,
                          Args... args)
{
  copyValue(fromIndex, toIndex, fromArray, toArray);
  copyValue(fromIndex, toIndex, args...);
}

template <typename T>
__device__ void copyValueIfSet(int fromIndex, int toIndex, bool flag,
                               T *fromArray, T *toArray)
{
  if (flag)
    toArray[toIndex] = fromArray[fromIndex];
}

template <typename T, typename... Args>
__device__ void copyValueIfSet(int fromIndex, int toIndex, bool flag,
                               T *fromArray, T *toArray, Args... args)
{
  copyValueIfSet(fromIndex, toIndex, flag, fromArray, toArray);
  copyValueIfSet(fromIndex, toIndex, flag, args...);
}

template <typename... Args>
__global__ void reorganizeKernel(int numValues, ReorganizeType reorganizeType,
                                 int *indices, int *flags, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
  {
    switch (reorganizeType)
    {
      case ReorganizeType::COPY_FROM_INDEX:
        copyValue(indices[tid], tid, args...);
        break;
      case ReorganizeType::COPY_TO_INDEX:
        copyValue(tid, indices[tid], args...);
        break;
      case ReorganizeType::CONDITIONAL_FROM_INDEX:
        copyValueIfSet(indices[tid], tid, 1 == flags[tid], args...);
        break;
      case ReorganizeType::CONDITIONAL_TO_INDEX:
        copyValueIfSet(tid, indices[tid], 1 == flags[tid], args...);
        break;
      default:
        break;
    }
  }
}

// Could be generalized to accept any comparable type, but I've been too lazy to
// do that...
__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values,
                                          double constant);
template <typename... Args>
__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values,
                                          double constant, Args... args)
{
  setFlagIfLessThanConstant(idx, flags, values, constant);
  setFlagIfLessThanConstant(idx, args...);
}

template <typename... Args>
__global__ void setFlagIfLessThanConstantKernel(int numValues, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
    setFlagIfLessThanConstant(tid, args...);
}

__device__ void setFlagIfGreaterThanConstant(int idx, int *flags,
                                             double *values, double constant);
template <typename... Args>
__device__ void setFlagIfGreaterThanConstant(int idx, int *flags,
                                             double *values, double constant,
                                             Args... args)
{
  setFlagIfGreaterThanConstant(idx, flags, values, constant);
  setFlagIfGreaterThanConstant(idx, args...);
}

template <typename... Args>
__global__ void setFlagIfGreaterThanConstantKernel(int numValues, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
    setFlagIfGreaterThanConstant(tid, args...);
}

__global__ void transformPositionsKernel(bool normalize, int numValues,
                                         dvec lbb, dvec tfr, double *x,
                                         double *y, double *z);

__device__ double getWrappedDistance(double x1, double x2, double maxDistance,
                                     bool shouldWrap);

__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance,
                                     bool shouldWrap, double *x);
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance,
                                     double minDistance, bool shouldWrap,
                                     double *x, double *useless);
template <typename... Args>
__forceinline__ __device__ double getDistanceSquared(
  int idx1, int idx2, double maxDistance, double minDistance, bool shouldWrap,
  double *x, double *useless, Args... args)
{
  double d = getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
  d += getDistanceSquared(idx1, idx2, args...);

  return d;
}
template <typename... Args>
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance,
                                     bool shouldWrap, double *x, Args... args)
{
  double d = getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
  d += getDistanceSquared(idx1, idx2, args...);

  return d;
}

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum);
__device__ double getWrappedCoordinate(double val1, double val2,
                                       double multiplier);
__device__ int getCellIdxFromPos(double x, double y, double z, dvec lbb,
                                 dvec tfr, ivec cellDim);
__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim);
__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim);

__device__ __host__ unsigned int encodeMorton2(unsigned int x, unsigned int y);
__device__ __host__ unsigned int encodeMorton3(unsigned int x, unsigned int y,
                                               unsigned int z);
__device__ __host__ unsigned int decodeMorton2x(unsigned int code);
__device__ __host__ unsigned int decodeMorton2y(unsigned int code);
__device__ __host__ unsigned int decodeMorton3x(unsigned int code);
__device__ __host__ unsigned int decodeMorton3y(unsigned int code);
__device__ __host__ unsigned int decodeMorton3z(unsigned int code);
__device__ __host__ unsigned int part1By1(unsigned int x);
__device__ __host__ unsigned int part1By2(unsigned int x);
__device__ __host__ unsigned int compact1By1(unsigned int x);
__device__ __host__ unsigned int compact1By2(unsigned int x);

template <typename... Args>
__device__ void comparePair(int idx1, int idx2, double *r, int *first,
                            int *second, Args... args)
{
  double maxDistance = r[idx1] > r[idx2] ? r[idx1] : r[idx2];
  maxDistance *= 2.5;
  maxDistance += (r[idx1] < r[idx2]) ? r[idx1] : r[idx2];
  if (getDistanceSquared(idx1, idx2, args...) < maxDistance * maxDistance)
  {
    // Set the smaller idx to idx1 and larger to idx2
    int id = idx1 > idx2 ? idx1 : idx2;
    idx1   = idx1 < idx2 ? idx1 : idx2;
    idx2   = id;

    id         = atomicAdd(&dNumPairs, 1);
    first[id]  = idx1;
    second[id] = idx2;
  }
}

__device__ void wrapAround(int idx, double *coordinate, double minValue,
                           double maxValue, int *wrapMultiplier,
                           int *wrapMultiplierPrev);
template <typename... Args>
__device__ void wrapAround(int idx, double *coordinate, double minValue,
                           double maxValue, int *wrapMultiplier,
                           int *wrapMultiplierPrev, Args... args)
{
  wrapAround(idx, coordinate, minValue, maxValue, wrapMultiplier,
             wrapMultiplierPrev);
  wrapAround(idx, args...);
}

__device__ void addVelocity(int idx1, int idx2, double multiplier,
                            double maxDistance, double minDistance,
                            bool shouldWrap, double *x, double *v);
template <typename... Args>
__device__ void addVelocity(int idx1, int idx2, double multiplier,
                            double maxDistance, double minDistance,
                            bool shouldWrap, double *x, double *v, Args... args)
{
  addVelocity(idx1, idx2, multiplier, maxDistance, minDistance, shouldWrap, x,
              v);
  addVelocity(idx1, idx2, multiplier, args...);
}

template <typename... Args>
__device__ void forceBetweenPair(int idx1, int idx2, double fZeroPerMuZero,
                                 double *r, Args... args)
{
  const double radii = r[idx1] + r[idx2];
  double multiplier  = getDistanceSquared(idx1, idx2, args...);
  if (radii * radii >= multiplier)
  {
    multiplier = sqrt(multiplier);
    multiplier = fZeroPerMuZero * (radii - multiplier) / (radii * multiplier);
    addVelocity(idx1, idx2, multiplier, args...);
  }
}


__device__ void addNeighborVelocity(int idx1, int idx2, double *sumOfVelocities,
                                    double *velocity);
template <typename... Args>
__device__ void addNeighborVelocity(int idx1, int idx2, double *sumOfVelocities,
                                    double *velocity, Args... args)
{
  addNeighborVelocity(idx1, idx2, sumOfVelocities, velocity);
  addNeighborVelocity(idx1, idx2, args...);
}

template <typename... Args>
__global__ void boundaryWrapKernel(int numValues, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
    wrapAround(tid, args...);
}

__global__ void calculateVolumes(double *r, double *volumes, int numValues);

__global__ void assignDataToBubbles(double *x, double *y, double *z,
                                    double *xPrd, double *yPrd, double *zPrd,
                                    double *r, double *w, int *aboveMinRadFlags,
                                    int *indices, ivec bubblesPerDim, dvec tfr,
                                    dvec lbb, double avgRad, double minRad,
                                    int numValues);

__global__ void assignBubblesToCells(double *x, double *y, double *z,
                                     int *cellIndices, int *bubbleIndices,
                                     dvec lbb, dvec tfr, ivec cellDim,
                                     int numValues);

template <typename... Args>
__global__ void neighborSearch(int neighborCellNumber, int numValues,
                               int numCells, int numMaxPairs, int *offsets,
                               int *sizes, int *first, int *second, double *r,
                               Args... args)
{
  const ivec idxVec(blockIdx.x, blockIdx.y, blockIdx.z);
  const ivec dimVec(gridDim.x, gridDim.y, gridDim.z);
  const int cellIdx2 = getNeighborCellIndex(idxVec, dimVec, neighborCellNumber);

  if (cellIdx2 >= 0)
  {
    const int cellIdx1 = get1DIdxFrom3DIdx(idxVec, dimVec);
    DEVICE_ASSERT(cellIdx1 < numCells, "Invalid cell index!");
    DEVICE_ASSERT(cellIdx2 < numCells, "Invalid cell index!");

    if (sizes[cellIdx1] == 0 || sizes[cellIdx2] == 0)
      return;

    // Self comparison only loops the upper triangle of values (n * (n - 1)) / 2
    // comparisons instead of n^2.
    if (cellIdx1 == cellIdx2)
    {
      const int size   = sizes[cellIdx1];
      const int offset = offsets[cellIdx1];
      for (int k = threadIdx.x; k < (size * (size - 1)) / 2; k += blockDim.x)
      {
        int idx1 =
          size - 2 -
          (int)floor(sqrt(-8.0 * k + 4 * size * (size - 1) - 7) * 0.5 - 0.5);
        const int idx2 = offset + k + idx1 + 1 - size * (size - 1) / 2 +
                         (size - idx1) * ((size - idx1) - 1) / 2;
        idx1 += offset;

        DEVICE_ASSERT(idx1 < numValues, "Invalid bubble index!");
        DEVICE_ASSERT(idx2 < numValues, "Invalid bubble index!");
        DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

        comparePair(idx1, idx2, r, first, second, args...);
        DEVICE_ASSERT(numMaxPairs > dNumPairs, "Too many neighbor indices!");
      }
    }
    else // Compare all values of one cell to all values of other cell,
         // resulting in n1 * n2
         // comparisons.
    {
      const int size1   = sizes[cellIdx1];
      const int size2   = sizes[cellIdx2];
      const int offset1 = offsets[cellIdx1];
      const int offset2 = offsets[cellIdx2];
      for (int k = threadIdx.x; k < size1 * size2; k += blockDim.x)
      {
        const int idx1 = offset1 + k / size2;
        const int idx2 = offset2 + k % size2;

        DEVICE_ASSERT(idx1 < numValues, "Invalid bubble index!");
        DEVICE_ASSERT(idx2 < numValues, "Invalid bubble index!");
        DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

        comparePair(idx1, idx2, r, first, second, args...);
        DEVICE_ASSERT(numMaxPairs > dNumPairs, "Too many neighbor indices!");
      }
    }
  }
}

template <typename... Args>
__global__ void velocityPairKernel(double fZeroPerMuZero, int *first,
                                   int *second, double *r, Args... args)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
       i += gridDim.x * blockDim.x)
    forceBetweenPair(first[i], second[i], fZeroPerMuZero, r, args...);
}

__global__ void velocityWallKernel(int numValues, double *r, double *x,
                                   double *y, double *z, double *vx, double *vy,
                                   double *vz, dvec lbb, dvec tfr,
                                   double fZeroPerMuZero, double dragCoeff);

template <typename... Args>
__global__ void neighborVelocityKernel(int *first, int *second,
                                       int *numNeighbors, Args... args)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
       i += gridDim.x * blockDim.x)
  {
    const int idx1 = first[i];
    const int idx2 = second[i];
    atomicAdd(&numNeighbors[idx1], 1);
    atomicAdd(&numNeighbors[idx2], 1);
    addNeighborVelocity(idx1, idx2, args...);
  }
}

__global__ void flowVelocityKernel(int numValues, int *numNeighbors,
                                   double *velX, double *velY, double *velZ,
                                   double *nVelX, double *nVelY, double *nVelZ,
                                   double *posX, double *posY, double *posZ,
                                   double *r, dvec flowVel, dvec flowTfr,
                                   dvec flowLbb);

template <typename... Args>
__global__ void potentialEnergyKernel(int numValues, int *first, int *second,
                                      double *r, double *energy, Args... args)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
       i += gridDim.x * blockDim.x)
  {
    const int idx1 = first[i];
    const int idx2 = second[i];
    double e =
      r[idx1] + r[idx2] - sqrt(getDistanceSquared(idx1, idx2, args...));
    if (e > 0)
    {
      e *= e;
      atomicAdd(&energy[idx1], e);
      atomicAdd(&energy[idx2], e);
    }
  }
}

template <typename... Args>
__global__ void gasExchangeKernel(int numValues, int *first, int *second,
                                  double *r, double *drdt, double *freeArea,
                                  Args... args)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
       i += gridDim.x * blockDim.x)
  {
    const int idx1 = first[i];
    const int idx2 = second[i];

    const double magnitude = sqrt(getDistanceSquared(idx1, idx2, args...));
    const double r1        = r[idx1];
    const double r2        = r[idx2];

    if (magnitude < r1 + r2)
    {
      double overlapArea = 0;

      if (magnitude < r1 || magnitude < r2)
      {
        overlapArea = r1 < r2 ? r1 : r2;
        overlapArea *= overlapArea;
      }
      else
      {
        overlapArea =
          0.5 * (r2 * r2 - r1 * r1 + magnitude * magnitude) / magnitude;
        overlapArea *= overlapArea;
        overlapArea = r2 * r2 - overlapArea;
        DEVICE_ASSERT(overlapArea > -0.0001, "Overlap area is negative!");
        overlapArea = overlapArea < 0 ? -overlapArea : overlapArea;
        DEVICE_ASSERT(overlapArea >= 0, "Overlap area is negative!");
      }
#if (NUM_DIM == 3)
      overlapArea *= CUBBLE_PI;
#else
      overlapArea = 2.0 * sqrt(overlapArea);
#endif
      atomicAdd(&freeArea[idx1], overlapArea);
      atomicAdd(&freeArea[idx2], overlapArea);

      overlapArea *= (1.0 / r2 - 1.0 / r1);

      atomicAdd(&drdt[idx1], overlapArea);
      atomicAdd(&drdt[idx2], -overlapArea);
    }
  }
}

__global__ void freeAreaKernel(int numValues, double *r, double *freeArea,
                               double *freeAreaPerRadius, double *area);

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r,
                                            double *freeArea, int numValues,
                                            double kappa, double kParam,
                                            double averageSurfaceAreaIn);

__global__ void addVolume(double *r, int numValues);

__global__ void calculateRedistributedGasVolume(double *volume, double *r,
                                                int *aboveMinRadFlags,
                                                int numValues);

__device__ void adamsBashforth(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fPrevious);
template <typename... Args>
__device__ void adamsBashforth(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fPrevious,
                               Args... args)
{
  adamsBashforth(idx, timeStep, yNext, y, f, fPrevious);
  adamsBashforth(idx, timeStep, args...);
}

template <typename... Args>
__global__ void predictKernel(int numValues, double timeStep, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
    adamsBashforth(tid, timeStep, args...);
}

__device__ double adamsMoulton(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fNext);
template <typename... Args>
__device__ double adamsMoulton(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fNext,
                               Args... args)
{
  double error1       = adamsMoulton(idx, timeStep, yNext, y, f, fNext);
  const double error2 = adamsMoulton(idx, timeStep, args...);
  error1              = error1 > error2 ? error1 : error2;

  return error1;
}

template <typename... Args>
__global__ void correctKernel(int numValues, double timeStep, double *errors,
                              Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
  {
    const double e = adamsMoulton(tid, timeStep, args...);
    errors[tid]    = e > errors[tid] ? e : errors[tid];
  }
}

__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f);
template <typename... Args>
__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f,
                               Args... args)
{
  eulerIntegrate(idx, timeStep, y, f);
  eulerIntegrate(idx, timeStep, args...);
}

template <typename... Args>
__global__ void eulerKernel(int numValues, double timeStep, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
    eulerIntegrate(tid, timeStep, args...);
}

__device__ double calculateDistanceFromStart(int idx, double *x, double *xPrev,
                                             double *xStart,
                                             int *wrapMultiplier,
                                             double interval);
template <typename... Args>
__device__ double calculateDistanceFromStart(int idx, double *x, double *xPrev,
                                             double *xStart,
                                             int *wrapMultiplier,
                                             double interval, Args... args)
{
  return calculateDistanceFromStart(idx, args...) +
         calculateDistanceFromStart(idx, x, xPrev, xStart, wrapMultiplier,
                                    interval);
}

__device__ double calculatePathLength(int idx, double *x, double *xPrev,
                                      double *xStart, int *wrapMultiplier,
                                      double interval);
template <typename... Args>
__device__ double calculatePathLength(int idx, double *x, double *xPrev,
                                      double *xStart, int *wrapMultiplier,
                                      double interval, Args... args)
{
  return calculatePathLength(idx, args...) +
         calculatePathLength(idx, x, xPrev, xStart, wrapMultiplier, interval);
}

template <typename... Args>
__global__ void pathLengthDistanceKernel(int numValues, double *pathLengths,
                                         double *pathLengthsPrev,
                                         double *squaredDistances, Args... args)
{
  const int tid = getGlobalTid();
  if (tid < numValues)
  {
    pathLengths[tid] =
      pathLengthsPrev[tid] + sqrt(calculatePathLength(tid, args...));
    squaredDistances[tid] = calculateDistanceFromStart(tid, args...);
  }
}

__device__ void test_vec_add(float *x, float *y, float *z, int N);

} // namespace cubble

