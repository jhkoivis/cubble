#include "CubWrapper.h"
#include "Env.h"
#include "Kernels.cuh"
#include "Macros.h"
#include "Vec.h"
#include "cub/cub/cub.cuh"
#include <array>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <iostream>
#include <nvToolsExt.h>
#include <sstream>
#include <vector>

namespace cubble
{
// Device double pointers
enum class DDP
{
  X,
  Y,
  Z,
  R,

  DXDT,
  DYDT,
  DZDT,
  DRDT,

  DXDTO,
  DYDTO,
  DZDTO,
  DRDTO,

  X0,
  Y0,
  Z0,

  PATH,
  DISTANCE,

  XP,
  YP,
  ZP,
  RP,

  DXDTP,
  DYDTP,
  DZDTP,
  DRDTP,

  ERROR,

  TEMP1,
  TEMP2,
  TEMP3,
  TEMP4,
  TEMP5,
  TEMP6,
  TEMP7,
  TEMP8,

  NUM_VALUES
};

// Device int pointers
enum class DIP
{
  FLAGS,

  WRAP_COUNT_X,
  WRAP_COUNT_Y,
  WRAP_COUNT_Z,

  WRAP_COUNT_XP,
  WRAP_COUNT_YP,
  WRAP_COUNT_ZP,

  PAIR1,
  PAIR2,

  TEMP1,
  TEMP2,

  NUM_VALUES
};

struct SimulationState
{
  double simulationTime  = 0.0;
  double energy1         = 0.0;
  double energy2         = 0.0;
  double maxBubbleRadius = 0.0;
  size_t integrationStep = 0;
  uint32_t numSnapshots  = 0;
  int numBubbles         = 0;
  int maxNumCells        = 0;
  int numPairs           = 0;

  // Host pointers to device global variables
  int *mbpc      = nullptr;
  int *np        = nullptr;
  double *dtfapr = nullptr;
  double *dtfa   = nullptr;
  double *dvm    = nullptr;
  double *dtv    = nullptr;
  double *dir    = nullptr;
  double *dta    = nullptr;
  double *dasai  = nullptr;

  // Device data
  double *pinnedDoubles = nullptr;
  double *deviceDoubles = nullptr;
  int *deviceInts       = nullptr;
  int *pinnedInts       = nullptr;
  uint32_t dataStride   = 0;
  uint32_t pairStride   = 0;
  uint64_t memReqD      = 0;
  uint64_t memReqI      = 0;

  std::array<double *, (uint64_t)DDP::NUM_VALUES> ddps;
  std::array<int *, (uint64_t)DIP::NUM_VALUES> dips;
};

struct Params
{
  SimulationState state;
  CubWrapper cw;
  Env env;

  cudaStream_t velocityStream;
  cudaStream_t gasExchangeStream;

  KernelSize pairKernelSize = KernelSize(dim3(256, 1, 1), dim3(128, 1, 1));
}

} // namespace cubble

namespace // anonymous
{
using namespace cubble;

void doBoundaryWrap(KernelSize ks, int sm, cudaStream_t stream, bool wrapX, bool wrapY, bool wrapZ, int numValues,
                    double *x, double *y, double *z, dvec lbb, dvec tfr, int *mulX, int *mulY, int *mulZ, int *mulOldX,
                    int *mulOldY, int *mulOldZ)
{
  if (wrapX && wrapY && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, y, lbb.y, tfr.y, mulY,
                  mulOldY, z, lbb.z, tfr.z, mulZ, mulOldZ);
  else if (wrapX && wrapY)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, y, lbb.y, tfr.y, mulY,
                  mulOldY);
  else if (wrapX && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX, z, lbb.z, tfr.z, mulZ,
                  mulOldZ);
  else if (wrapY && wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, y, lbb.y, tfr.y, mulY, mulOldY, z, lbb.z, tfr.z, mulZ,
                  mulOldZ);
  else if (wrapX)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, x, lbb.x, tfr.x, mulX, mulOldX);
  else if (wrapY)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, y, lbb.y, tfr.y, mulY, mulOldY);
  else if (wrapZ)
    KERNEL_LAUNCH(boundaryWrapKernel, ks, sm, stream, numValues, z, lbb.z, tfr.z, mulZ, mulOldZ);
}

void doWallVelocity(KernelSize ks, int sm, cudaStream_t stream, bool doX, bool doY, bool doZ, int numValues, int *first,
                    int *second, double *r, double *x, double *y, double *z, double *dxdt, double *dydt, double *dzdt,
                    dvec lbb, dvec tfr, Env properties)
{
  dvec interval = tfr - lbb;
  if (doX && doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX && doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doX && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doY && doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.y, lbb.y, !doY, y, dydt, interval.z, lbb.z, !doZ, z, dzdt);
  }
  else if (doX)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.x, lbb.x, !doX, x, dxdt);
  }
  else if (doY)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.y, lbb.y, !doY, y, dydt);
  }
  else if (doZ)
  {
    KERNEL_LAUNCH(velocityWallKernel, ks, sm, stream, numValues, params.env.getFZeroPerMuZero(), first, second, r,
                  interval.z, lbb.z, !doZ, z, dzdt);
  }
}

void startProfiling(bool start)
{
  if (start)
    cudaProfilerStart();
}

void stopProfiling(bool stop, bool &continueIntegration)
{
  if (stop)
  {
    cudaProfilerStop();
    continueIntegration = false;
  }
}

void reserveMemory(Params &params)
{
  // Reserve pinned memory
  CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&params.state.pinnedDoubles), 3 * sizeof(double)));
  CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&params.state.pinnedInts), 1 * sizeof(int)));

  // Calculate the length of 'rows'. Will be divisible by 32, as that's the warp size.
  params.state.dataStride =
    params.state.numBubbles + !!(params.state.numBubbles % 32) * (32 - params.state.numBubbles % 32);

  // Doubles
  params.state.memReqD = sizeof(double) * (uint64_t)params.state.dataStride * (uint64_t)DDP::NUM_VALUES;
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.state.deviceDoubles), params.state.memReqD));

  for (uint32_t i = 0; i < (uint32_t)DDP::NUM_VALUES; ++i)
    params.state.ddps[i] = params.state.deviceDoubles + i * params.state.dataStride;

  // Integers
  const uint32_t avgNumNeighbors = 32;
  params.state.pairStride        = avgNumNeighbors * params.state.dataStride;

  params.state.memReqI = sizeof(int) * (uint64_t)params.state.dataStride *
                         ((uint64_t)DIP::PAIR1 + avgNumNeighbors * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1));
  CUDA_ASSERT(cudaMalloc(reinterpret_cast<void **>(&params.state.deviceInts), params.state.memReqI));

  for (uint32_t i = 0; i < (uint32_t)DIP::PAIR2; ++i)
    params.state.dips[i] = params.state.deviceInts + i * params.state.dataStride;

  uint32_t j = 0;
  for (uint32_t i = (uint32_t)DIP::PAIR2; i < (uint32_t)DIP::NUM_VALUES; ++i)
    params.state.dips[i] = params.state.dips[(uint32_t)DIP::PAIR1] + avgNumNeighbors * ++j * params.state.dataStride;
}

dim3 getGridSize(Params &params)
{
  const int totalNumCells = std::ceil((float)params.state.numBubbles / params.env.getNumBubblesPerCell());
  dvec interval           = params.env.getTfr() - params.env.getLbb();
  interval                = interval / interval.x;
  float nx                = (float)totalNumCells / interval.y;
  if (NUM_DIM == 3)
    nx = std::cbrt(nx / interval.z);
  else
  {
    nx         = std::sqrt(nx);
    interval.z = 0;
  }

  ivec grid = (nx * interval).floor() + 1;
  assert(grid.x > 0);
  assert(grid.y > 0);
  assert(grid.z > 0);

  return dim3(grid.x, grid.y, grid.z);
}

void updateCellsAndNeighbors(Params &params)
{
  dim3 gridSize = getGridSize(params);
  const ivec cellDim(gridSize.x, gridSize.y, gridSize.z);

  int *offsets             = params.state.dips[(uint32_t)DIP::PAIR1];
  int *sizes               = params.state.dips[(uint32_t)DIP::PAIR1] + params.state.maxNumCells;
  int *cellIndices         = params.state.dips[(uint32_t)DIP::TEMP1] + 0 * params.state.dataStride;
  int *bubbleIndices       = params.state.dips[(uint32_t)DIP::TEMP1] + 1 * params.state.dataStride;
  int *sortedCellIndices   = params.state.dips[(uint32_t)DIP::TEMP1] + 2 * params.state.dataStride;
  int *sortedBubbleIndices = params.state.dips[(uint32_t)DIP::TEMP1] + 3 * params.state.dataStride;

  const size_t resetBytes = sizeof(int) * params.state.pairStride * ((uint64_t)DIP::NUM_VALUES - (uint64_t)DIP::PAIR1);
  CUDA_CALL(cudaMemset(params.state.dips[(uint32_t)DIP::PAIR1], 0, resetBytes));

  KernelSize kernelSize(128, params.state.numBubbles);

  KERNEL_LAUNCH(assignBubblesToCells, params.state.pairKernelSize, 0, 0, params.state.ddps[(uint32_t)DDP::X],
                params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::Z], cellIndices, bubbleIndices,
                params.env.getLbb(), params.env.getTfr(), cellDim, params.state.numBubbles);

  params.cw.sortPairs<int, int>(&cub::DeviceRadixSort::SortPairs, const_cast<const int *>(cellIndices),
                                sortedCellIndices, const_cast<const int *>(bubbleIndices), sortedBubbleIndices,
                                params.state.numBubbles);

  params.cw.histogram<int *, int, int, int>(&cub::DeviceHistogram::HistogramEven, cellIndices, sizes,
                                            params.state.maxNumCells + 1, 0, params.state.maxNumCells,
                                            params.state.numBubbles);

  params.cw.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, sizes, offsets, params.state.maxNumCells);

  KERNEL_LAUNCH(
    reorganizeKernel, kernelSize, 0, 0, params.state.numBubbles, ReorganizeType::COPY_FROM_INDEX, sortedBubbleIndices,
    sortedBubbleIndices, params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::XP],
    params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::Z],
    params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::RP],
    params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTP],
    params.state.ddps[(uint32_t)DDP::DYDT], params.state.ddps[(uint32_t)DDP::DYDTP],
    params.state.ddps[(uint32_t)DDP::DZDT], params.state.ddps[(uint32_t)DDP::DZDTP],
    params.state.ddps[(uint32_t)DDP::DRDT], params.state.ddps[(uint32_t)DDP::DRDTP],
    params.state.ddps[(uint32_t)DDP::DXDTO], params.state.ddps[(uint32_t)DDP::ERROR],
    params.state.ddps[(uint32_t)DDP::DYDTO], params.state.ddps[(uint32_t)DDP::TEMP1],
    params.state.ddps[(uint32_t)DDP::DZDTO], params.state.ddps[(uint32_t)DDP::TEMP2],
    params.state.ddps[(uint32_t)DDP::DRDTO], params.state.ddps[(uint32_t)DDP::TEMP3],
    params.state.ddps[(uint32_t)DDP::X0], params.state.ddps[(uint32_t)DDP::TEMP4], params.state.ddps[(uint32_t)DDP::Y0],
    params.state.ddps[(uint32_t)DDP::TEMP5], params.state.ddps[(uint32_t)DDP::Z0],
    params.state.ddps[(uint32_t)DDP::TEMP6], params.state.ddps[(uint32_t)DDP::PATH],
    params.state.ddps[(uint32_t)DDP::TEMP7], params.state.ddps[(uint32_t)DDP::DISTANCE],
    params.state.ddps[(uint32_t)DDP::TEMP8], params.state.dips[(uint32_t)DIP::WRAP_COUNT_X],
    params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
    params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z],
    params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.state.ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(params.state.ddps[(uint32_t)DDP::XP]), params.state.memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  dvec interval = params.env.getTfr() - params.env.getLbb();

  kernelSize.block = dim3(128, 1, 1);
  kernelSize.grid  = gridSize;

  CUDA_CALL(cudaMemset(params.state.np, 0, sizeof(int)));

  const double maxDistance = 1.5 * params.state.maxBubbleRadius;

  for (int i = 0; i < CUBBLE_NUM_NEIGHBORS + 1; ++i)
  {
    cudaStream_t stream = (i % 2) ? params.state.velocityStream : params.state.gasExchangeStream;
    if (NUM_DIM == 3)
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, params.state.numBubbles, params.state.maxNumCells,
                    (int)params.state.pairStride, maxDistance, offsets, sizes, params.state.dips[(uint32_t)DIP::TEMP1],
                    params.state.dips[(uint32_t)DIP::TEMP2], params.state.ddps[(uint32_t)DDP::R], interval.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                    params.state.ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::Z]);
    else
      KERNEL_LAUNCH(neighborSearch, kernelSize, 0, stream, i, params.state.numBubbles, params.state.maxNumCells,
                    (int)params.state.pairStride, maxDistance, offsets, sizes, params.state.dips[(uint32_t)DIP::TEMP1],
                    params.state.dips[(uint32_t)DIP::TEMP2], params.state.ddps[(uint32_t)DDP::R], interval.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                    params.state.ddps[(uint32_t)DDP::Y]);
  }

  CUDA_CALL(
    cudaMemcpy(static_cast<void *>(params.state.pinnedInts), params.state.np, sizeof(int), cudaMemcpyDeviceToHost));
  params.state.numPairs = params.state.pinnedInts[0];
  params.cw.sortPairs<int, int>(
    &cub::DeviceRadixSort::SortPairs, const_cast<const int *>(params.state.dips[(uint32_t)DIP::TEMP1]),
    params.state.dips[(uint32_t)DIP::PAIR1], const_cast<const int *>(params.state.dips[(uint32_t)DIP::TEMP2]),
    params.state.dips[(uint32_t)DIP::PAIR2], params.state.numPairs);
}

void deleteSmallBubbles(Params &params, int numBubblesAboveMinRad)
{
  NVTX_RANGE_PUSH_A("BubbleRemoval");
  KernelSize kernelSize(128, params.state.numBubbles);

  CUDA_CALL(cudaMemset(static_cast<void *>(params.state.dvm), 0, sizeof(double)));
  KERNEL_LAUNCH(calculateRedistributedGasVolume, kernelSize, 0, 0, params.state.ddps[(uint32_t)DDP::TEMP1],
                params.state.ddps[(uint32_t)DDP::R], params.state.dips[(uint32_t)DIP::FLAGS], params.state.numBubbles);

  params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP1],
                                                     params.state.dtv, params.state.numBubbles);

  int *newIdx = params.state.dips[(uint32_t)DIP::TEMP1];
  params.cw.scan<int *, int *>(&cub::DeviceScan::ExclusiveSum, params.state.dips[(uint32_t)DIP::FLAGS], newIdx,
                               params.state.numBubbles);

  KERNEL_LAUNCH(
    reorganizeKernel, kernelSize, 0, 0, params.state.numBubbles, ReorganizeType::CONDITIONAL_TO_INDEX, newIdx,
    params.state.dips[(uint32_t)DIP::FLAGS], params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::XP],
    params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::Z],
    params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::RP],
    params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTP],
    params.state.ddps[(uint32_t)DDP::DYDT], params.state.ddps[(uint32_t)DDP::DYDTP],
    params.state.ddps[(uint32_t)DDP::DZDT], params.state.ddps[(uint32_t)DDP::DZDTP],
    params.state.ddps[(uint32_t)DDP::DRDT], params.state.ddps[(uint32_t)DDP::DRDTP],
    params.state.ddps[(uint32_t)DDP::DXDTO], params.state.ddps[(uint32_t)DDP::ERROR],
    params.state.ddps[(uint32_t)DDP::DYDTO], params.state.ddps[(uint32_t)DDP::TEMP1],
    params.state.ddps[(uint32_t)DDP::DZDTO], params.state.ddps[(uint32_t)DDP::TEMP2],
    params.state.ddps[(uint32_t)DDP::DRDTO], params.state.ddps[(uint32_t)DDP::TEMP3],
    params.state.ddps[(uint32_t)DDP::X0], params.state.ddps[(uint32_t)DDP::TEMP4], params.state.ddps[(uint32_t)DDP::Y0],
    params.state.ddps[(uint32_t)DDP::TEMP5], params.state.ddps[(uint32_t)DDP::Z0],
    params.state.ddps[(uint32_t)DDP::TEMP6], params.state.ddps[(uint32_t)DDP::PATH],
    params.state.ddps[(uint32_t)DDP::TEMP7], params.state.ddps[(uint32_t)DDP::DISTANCE],
    params.state.ddps[(uint32_t)DDP::TEMP8], params.state.dips[(uint32_t)DIP::WRAP_COUNT_X],
    params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
    params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z],
    params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.state.ddps[(uint32_t)DDP::X]),
                            static_cast<void *>(params.state.ddps[(uint32_t)DDP::XP]), params.state.memReqD / 2,
                            cudaMemcpyDeviceToDevice));

  params.state.numBubbles = numBubblesAboveMinRad;
  KERNEL_LAUNCH(addVolume, kernelSize, 0, 0, params.state.ddps[(uint32_t)DDP::R], params.state.numBubbles);

  NVTX_RANGE_POP();
}

void setup(Params &params)
{
  // First calculate the size of the box and the starting number of bubbles
  dvec relDim        = params.env.getBoxRelativeDimensions();
  relDim             = relDim / relDim.x;
  const float d      = 2 * params.env.getAvgRad();
  float x            = params.env.getNumBubbles() * d * d / relDim.y;
  ivec bubblesPerDim = ivec(0, 0, 0);

  if (NUM_DIM == 3)
  {
    x                       = x * d / relDim.z;
    x                       = std::cbrt(x);
    relDim                  = relDim * x;
    bubblesPerDim           = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), std::ceil(relDim.z / d));
    params.state.numBubbles = bubblesPerDim.x * bubblesPerDim.y * bubblesPerDim.z;
  }
  else
  {
    x                       = std::sqrt(x);
    relDim                  = relDim * x;
    bubblesPerDim           = ivec(std::ceil(relDim.x / d), std::ceil(relDim.y / d), 0);
    params.state.numBubbles = bubblesPerDim.x * bubblesPerDim.y;
  }

  params.env.setTfr(d * bubblesPerDim.asType<double>() + params.env.getLbb());
  dvec interval = params.env.getTfr() - params.env.getLbb();
  params.env.setFlowTfr(interval * params.env.getFlowTfr() + params.env.getLbb());
  params.env.setFlowLbb(interval * params.env.getFlowLbb() + params.env.getLbb());

  // Determine the maximum number of Morton numbers for the simulation box
  dim3 gridDim         = getGridSize(params);
  const int maxGridDim = gridDim.x > gridDim.y ? (gridDim.x > gridDim.z ? gridDim.x : gridDim.z)
                                               : (gridDim.y > gridDim.z ? gridDim.y : gridDim.z);
  int maxNumCells = 1;
  while (maxNumCells < maxGridDim)
    maxNumCells = maxNumCells << 1;

  if (NUM_DIM == 3)
    maxNumCells = maxNumCells * maxNumCells * maxNumCells;
  else
    maxNumCells = maxNumCells * maxNumCells;

  params.state.maxNumCells = maxNumCells;

  std::cout << "Maximum (theoretical) number of cells: " << params.state.maxNumCells
            << ", actual grid dimensions: " << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << std::endl;

  // Reserve memory for data
  reserveMemory(params);
  std::cout << "Memory requirement for data:\n\tdouble: " << params.state.memReqD
            << " bytes\n\tint: " << params.state.memReqI << " bytes" << std::endl;

  // Get some device global symbol addresses to host pointers.
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dtfa), dTotalFreeArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dtfapr), dTotalFreeAreaPerRadius));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.mbpc), dMaxBubblesPerCell));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dvm), dVolumeMultiplier));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dtv), dTotalVolume));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.np), dNumPairs));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dir), dInvRho));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dta), dTotalArea));
  CUDA_ASSERT(cudaGetSymbolAddress(reinterpret_cast<void **>(&params.state.dasai), dAverageSurfaceAreaIn));

  // Streams
  CUDA_ASSERT(cudaStreamCreate(&params.state.velocityStream));
  CUDA_ASSERT(cudaStreamCreate(&params.state.gasExchangeStream));

  printRelevantInfoOfCurrentDevice();

  KernelSize kernelSize(128, params.state.numBubbles);

  std::cout << "Starting to generate data for bubbles." << std::endl;
  double timeStep        = params.env.getTimeStep();
  const int rngSeed      = params.env.getRngSeed();
  const double avgRad    = params.env.getAvgRad();
  const double stdDevRad = params.env.getStdDevRad();
  const dvec tfr         = params.env.getTfr();
  const dvec lbb         = params.env.getLbb();
  interval               = tfr - lbb;

  curandGenerator_t generator;
  CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, rngSeed));
  if (NUM_DIM == 3)
    CURAND_CALL(curandGenerateUniformDouble(generator, params.state.ddps[(uint32_t)DDP::Z], params.state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, params.state.ddps[(uint32_t)DDP::X], params.state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, params.state.ddps[(uint32_t)DDP::Y], params.state.numBubbles));
  CURAND_CALL(curandGenerateUniformDouble(generator, params.state.ddps[(uint32_t)DDP::RP], params.state.numBubbles));
  CURAND_CALL(curandGenerateNormalDouble(generator, params.state.ddps[(uint32_t)DDP::R], params.state.numBubbles,
                                         avgRad, stdDevRad));
  CURAND_CALL(curandDestroyGenerator(generator));

  KERNEL_LAUNCH(assignDataToBubbles, kernelSize, 0, 0, params.state.ddps[(uint32_t)DDP::X],
                params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::Z],
                params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::R],
                params.state.ddps[(uint32_t)DDP::RP], params.state.dips[(uint32_t)DIP::FLAGS], bubblesPerDim, tfr, lbb,
                avgRad, params.env.getMinRad(), params.state.numBubbles);

  params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::RP],
                                                     params.state.dasai, params.state.numBubbles, 0);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.state.ddps[(uint32_t)DDP::RP]),
                            static_cast<void *>(params.state.ddps[(uint32_t)DDP::R]),
                            sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice, 0));

  // Delete small bubbles, if any
  const int numBubblesAboveMinRad = params.cw.reduce<int, int *, int *>(
    &cub::DeviceReduce::Sum, params.state.dips[(uint32_t)DIP::FLAGS], params.state.numBubbles);
  if (numBubblesAboveMinRad < params.state.numBubbles)
    deleteSmallBubbles(params, numBubblesAboveMinRad);

  params.state.maxBubbleRadius = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Max, params.state.ddps[(uint32_t)DDP::R], params.state.numBubbles);

  updateCellsAndNeighbors(params);

  // Calculate some initial values which are needed
  // for the two-step Adams-Bashforth-Moulton prEdictor-corrector method
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles, params.state.ddps[(uint32_t)DDP::DXDTO],
                params.state.ddps[(uint32_t)DDP::DYDTO], params.state.ddps[(uint32_t)DDP::DZDTO],
                params.state.ddps[(uint32_t)DDP::DRDTO], params.state.ddps[(uint32_t)DDP::DISTANCE],
                params.state.ddps[(uint32_t)DDP::PATH]);

  std::cout << "Calculating some initial values as a part of setup." << std::endl;

  if (NUM_DIM == 3)
  {
    KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, 0, params.env.getFZeroPerMuZero(),
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], interval.x, lbb.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDTO], interval.y, lbb.y,
                  PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDTO], interval.z,
                  lbb.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::Z], params.state.ddps[(uint32_t)DDP::DZDTO]);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep, params.state.ddps[(uint32_t)DDP::X],
                  params.state.ddps[(uint32_t)DDP::DXDTO], params.state.ddps[(uint32_t)DDP::Y],
                  params.state.ddps[(uint32_t)DDP::DYDTO], params.state.ddps[(uint32_t)DDP::Z],
                  params.state.ddps[(uint32_t)DDP::DZDTO]);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, params.state.numBubbles,
                   params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                   params.state.ddps[(uint32_t)DDP::ZP], lbb, tfr, params.state.dips[(uint32_t)DIP::WRAP_COUNT_X],
                   params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                   params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                   params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles, params.state.ddps[(uint32_t)DDP::DXDTO],
                  params.state.ddps[(uint32_t)DDP::DYDTO], params.state.ddps[(uint32_t)DDP::DZDTO],
                  params.state.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, 0, params.env.getFZeroPerMuZero(),
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], interval.x, lbb.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDTO], interval.y, lbb.y,
                  PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDTO], interval.z,
                  lbb.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::Z], params.state.ddps[(uint32_t)DDP::DZDTO]);
  }
  else
  {
    KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, 0, params.env.getFZeroPerMuZero(),
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], interval.x, lbb.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDTO], interval.y, lbb.y,
                  PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDTO]);

    KERNEL_LAUNCH(eulerKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep, params.state.ddps[(uint32_t)DDP::X],
                  params.state.ddps[(uint32_t)DDP::DXDTO], params.state.ddps[(uint32_t)DDP::Y],
                  params.state.ddps[(uint32_t)DDP::DYDTO]);

    doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, false, params.state.numBubbles,
                   params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                   params.state.ddps[(uint32_t)DDP::ZP], lbb, tfr, params.state.dips[(uint32_t)DIP::WRAP_COUNT_X],
                   params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                   params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                   params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles, params.state.ddps[(uint32_t)DDP::DXDTO],
                  params.state.ddps[(uint32_t)DDP::DYDTO], params.state.ddps[(uint32_t)DDP::DRDTO]);

    KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, 0, params.env.getFZeroPerMuZero(),
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], interval.x, lbb.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDTO], interval.y, lbb.y,
                  PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDTO]);
  }
}

void saveSnapshotToFile(Params &params)
{
  std::stringstream ss;
  ss << params.env.getSnapshotFilename() << ".csv." << params.state.numSnapshots;
  std::ofstream file(ss.str().c_str(), std::ios::out);
  if (file.is_open())
  {
    std::vector<double> hostData;
    const size_t numComp = 17;
    hostData.resize(params.state.dataStride * numComp);
    CUDA_CALL(cudaMemcpy(hostData.data(), params.state.deviceDoubles,
                         sizeof(double) * numComp * params.state.dataStride, cudaMemcpyDeviceToHost));

    file << "x,y,z,r,vx,vy,vz,path,dist\n";
    for (size_t i = 0; i < (size_t)params.state.numBubbles; ++i)
    {
      file << hostData[i + 0 * params.state.dataStride];
      file << ",";
      file << hostData[i + 1 * params.state.dataStride];
      file << ",";
      file << hostData[i + 2 * params.state.dataStride];
      file << ",";
      file << hostData[i + 3 * params.state.dataStride];
      file << ",";
      file << hostData[i + 4 * params.state.dataStride];
      file << ",";
      file << hostData[i + 5 * params.state.dataStride];
      file << ",";
      file << hostData[i + 6 * params.state.dataStride];
      file << ",";
      file << hostData[i + 15 * params.state.dataStride];
      file << ",";
      file << hostData[i + 16 * params.state.dataStride];
      file << "\n";
    }

    ++params.state.numSnapshots;
  }
}

double stabilize(Params &params)
{
  // This function integrates only the positions of the bubbles.
  // Gas exchange is not used. This is used for equilibrating the foam.

  KernelSize kernelSize(128, params.state.numBubbles);

  const dvec tfr      = params.env.getTfr();
  const dvec lbb      = params.env.getLbb();
  const dvec interval = tfr - lbb;
  double elapsedTime  = 0.0;
  double timeStep     = params.env.getTimeStep();
  double error        = 100000;

  // Energy before stabilization
  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles, params.state.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y],
                  interval.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y]);

  params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP4],
                                                     params.state.dtfapr, params.state.numBubbles);
  CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(&params.state.pinnedDoubles[1]),
                            static_cast<void *>(params.state.dtfapr), sizeof(double), cudaMemcpyDeviceToHost, 0));

  for (int i = 0; i < params.env.getNumStepsToRelax(); ++i)
  {
    do
    {
      if (NUM_DIM == 3)
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles,
                      params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::DYDTP],
                      params.state.ddps[(uint32_t)DDP::DZDTP], params.state.ddps[(uint32_t)DDP::ERROR],
                      params.state.ddps[(uint32_t)DDP::TEMP1], params.state.ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep,
                      params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::X],
                      params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTO],
                      params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::Y],
                      params.state.ddps[(uint32_t)DDP::DYDT], params.state.ddps[(uint32_t)DDP::DYDTO],
                      params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::Z],
                      params.state.ddps[(uint32_t)DDP::DZDT], params.state.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, 0, params.env.getFZeroPerMuZero(),
                      params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                      params.state.ddps[(uint32_t)DDP::RP], interval.x, lbb.x, PBC_X == 1,
                      params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::DXDTP], interval.y, lbb.y,
                      PBC_Y == 1, params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::DYDTP],
                      interval.z, lbb.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::ZP],
                      params.state.ddps[(uint32_t)DDP::DZDTP]);

        doWallVelocity(params.state.pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, params.state.numBubbles,
                       params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                       params.state.ddps[(uint32_t)DDP::RP], params.state.ddps[(uint32_t)DDP::XP],
                       params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::ZP],
                       params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::DYDTP],
                       params.state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr, properties);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep,
                      params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::XP],
                      params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDT],
                      params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::YP],
                      params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDT],
                      params.state.ddps[(uint32_t)DDP::DYDTP], params.state.ddps[(uint32_t)DDP::ZP],
                      params.state.ddps[(uint32_t)DDP::Z], params.state.ddps[(uint32_t)DDP::DZDT],
                      params.state.ddps[(uint32_t)DDP::DZDTP]);
      }
      else // Two dimensional case
      {
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles,
                      params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::DYDTP],
                      params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::TEMP1],
                      params.state.ddps[(uint32_t)DDP::TEMP2]);

        KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep,
                      params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::X],
                      params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTO],
                      params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::Y],
                      params.state.ddps[(uint32_t)DDP::DYDT], params.state.ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, 0, params.env.getFZeroPerMuZero(),
                      params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                      params.state.ddps[(uint32_t)DDP::RP], interval.x, lbb.x, PBC_X == 1,
                      params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::DXDTP], interval.y, lbb.y,
                      PBC_Y == 1, params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::DYDTP]);

        doWallVelocity(params.state.pairKernelSize, 0, 0, PBC_X == 0, PBC_Y == 0, PBC_Z == 0, params.state.numBubbles,
                       params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                       params.state.ddps[(uint32_t)DDP::RP], params.state.ddps[(uint32_t)DDP::XP],
                       params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::ZP],
                       params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::DYDTP],
                       params.state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr, properties);

        KERNEL_LAUNCH(correctKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep,
                      params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::XP],
                      params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDT],
                      params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::YP],
                      params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDT],
                      params.state.ddps[(uint32_t)DDP::DYDTP]);
      }

      doBoundaryWrap(kernelSize, 0, 0, PBC_X == 1, PBC_Y == 1, PBC_Z == 1, params.state.numBubbles,
                     params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                     params.state.ddps[(uint32_t)DDP::ZP], lbb, tfr, params.state.dips[(uint32_t)DIP::WRAP_COUNT_X],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Error
      error = params.cw.reduce<double, double *, double *>(
        &cub::DeviceReduce::Max, params.state.ddps[(uint32_t)DDP::ERROR], params.state.numBubbles);

      if (error < params.env.getErrorTolerance() && timeStep < 0.1)
        timeStep *= 1.9;
      else if (error > params.env.getErrorTolerance())
        timeStep *= 0.5;

    } while (error > params.env.getErrorTolerance());

    // Update the current values with the calculated predictions
    const size_t numBytesToCopy = 3 * sizeof(double) * params.state.dataStride;
    CUDA_CALL(cudaMemcpyAsync(params.state.ddps[(uint32_t)DDP::DXDTO], params.state.ddps[(uint32_t)DDP::DXDT],
                              numBytesToCopy, cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::XP], numBytesToCopy,
                              cudaMemcpyDeviceToDevice, 0));
    CUDA_CALL(cudaMemcpyAsync(params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTP],
                              numBytesToCopy, cudaMemcpyDeviceToDevice, 0));

    params.env.setTimeStep(timeStep);
    elapsedTime += timeStep;

    if (i % 5000 == 0)
      updateCellsAndNeighbors(params);
  }

  // Energy after stabilization
  params.state.energy1 = params.state.pinnedDoubles[1];

  KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles, params.state.ddps[(uint32_t)DDP::TEMP4]);

  if (NUM_DIM == 3)
    KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y],
                  interval.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::Z]);
  else
    KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                  params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                  params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x, PBC_X == 1,
                  params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1, params.state.ddps[(uint32_t)DDP::Y]);

  params.state.energy2 = params.cw.reduce<double, double *, double *>(
    &cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP4], params.state.numBubbles);

  return elapsedTime;
}

bool integrate(Params &params)
{
  NVTX_RANGE_PUSH_A("Integration function");
  KernelSize kernelSize(128, params.state.numBubbles);

  const dvec tfr        = params.env.getTfr();
  const dvec lbb        = params.env.getLbb();
  const dvec interval   = tfr - lbb;
  double timeStep       = params.env.getTimeStep();
  double error          = 100000;
  uint32_t numLoopsDone = 0;

  do
  {
    NVTX_RANGE_PUSH_A("Integration step");

    if (NUM_DIM == 3)
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles,
                    params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::DYDTP],
                    params.state.ddps[(uint32_t)DDP::DZDTP], params.state.ddps[(uint32_t)DDP::DRDTP],
                    params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::TEMP1],
                    params.state.ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep,
                    params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::X],
                    params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTO],
                    params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::Y],
                    params.state.ddps[(uint32_t)DDP::DYDT], params.state.ddps[(uint32_t)DDP::DYDTO],
                    params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::Z],
                    params.state.ddps[(uint32_t)DDP::DZDT], params.state.ddps[(uint32_t)DDP::DZDTO],
                    params.state.ddps[(uint32_t)DDP::RP], params.state.ddps[(uint32_t)DDP::R],
                    params.state.ddps[(uint32_t)DDP::DRDT], params.state.ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, params.state.velocityStream,
                    params.env.getFZeroPerMuZero(), params.state.dips[(uint32_t)DIP::PAIR1],
                    params.state.dips[(uint32_t)DIP::PAIR2], params.state.ddps[(uint32_t)DDP::RP], interval.x, lbb.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::DXDTP],
                    interval.y, lbb.y, PBC_Y == 1, params.state.ddps[(uint32_t)DDP::YP],
                    params.state.ddps[(uint32_t)DDP::DYDTP], interval.z, lbb.z, PBC_Z == 1,
                    params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::DZDTP]);
      // Wall velocity
      doWallVelocity(params.state.pairKernelSize, 0, params.state.velocityStream, PBC_X == 0, PBC_Y == 0, PBC_Z == 0,
                     params.state.numBubbles, params.state.dips[(uint32_t)DIP::PAIR1],
                     params.state.dips[(uint32_t)DIP::PAIR2], params.state.ddps[(uint32_t)DDP::RP],
                     params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                     params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::DXDTP],
                     params.state.ddps[(uint32_t)DDP::DYDTP], params.state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr,
                     properties);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(params.state.dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * params.state.pairStride));
        int *numNeighbors = params.state.dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, params.state.pairKernelSize, 0, params.state.velocityStream,
                      params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      params.state.ddps[(uint32_t)DDP::TEMP1], params.state.ddps[(uint32_t)DDP::DXDTO],
                      params.state.ddps[(uint32_t)DDP::TEMP2], params.state.ddps[(uint32_t)DDP::DYDTO],
                      params.state.ddps[(uint32_t)DDP::TEMP3], params.state.ddps[(uint32_t)DDP::DZDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, params.state.pairKernelSize, 0, params.state.velocityStream,
                      params.state.numBubbles, numNeighbors, params.state.ddps[(uint32_t)DDP::DXDTP],
                      params.state.ddps[(uint32_t)DDP::DYDTP], params.state.ddps[(uint32_t)DDP::DZDTP],
                      params.state.ddps[(uint32_t)DDP::TEMP1], params.state.ddps[(uint32_t)DDP::TEMP2],
                      params.state.ddps[(uint32_t)DDP::TEMP3], params.state.ddps[(uint32_t)DDP::XP],
                      params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::ZP],
                      params.env.getFlowVel(), params.env.getFlowTfr(), params.env.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, params.state.velocityStream, params.state.numBubbles, timeStep,
                    params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::XP],
                    params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDT],
                    params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::YP],
                    params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDT],
                    params.state.ddps[(uint32_t)DDP::DYDTP], params.state.ddps[(uint32_t)DDP::ZP],
                    params.state.ddps[(uint32_t)DDP::Z], params.state.ddps[(uint32_t)DDP::DZDT],
                    params.state.ddps[(uint32_t)DDP::DZDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, params.state.velocityStream, params.state.numBubbles,
                    params.state.ddps[(uint32_t)DDP::TEMP4], params.state.ddps[(uint32_t)DDP::PATH],
                    params.state.ddps[(uint32_t)DDP::DISTANCE], params.state.ddps[(uint32_t)DDP::XP],
                    params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::X0],
                    params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], interval.x, params.state.ddps[(uint32_t)DDP::YP],
                    params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::Y0],
                    params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP], interval.y, params.state.ddps[(uint32_t)DDP::ZP],
                    params.state.ddps[(uint32_t)DDP::Z], params.state.ddps[(uint32_t)DDP::Z0],
                    params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP], interval.z);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, params.state.velocityStream, PBC_X == 1, PBC_Y == 1, PBC_Z == 1,
                     params.state.numBubbles, params.state.ddps[(uint32_t)DDP::XP],
                     params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::ZP], lbb, tfr,
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_X], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z], params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, params.state.pairKernelSize, 0, params.state.gasExchangeStream,
                    params.state.numBubbles, params.state.dips[(uint32_t)DIP::PAIR1],
                    params.state.dips[(uint32_t)DIP::PAIR2], params.state.ddps[(uint32_t)DDP::RP],
                    params.state.ddps[(uint32_t)DDP::DRDTP], params.state.ddps[(uint32_t)DDP::TEMP1], interval.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::XP], interval.y, PBC_Y == 1,
                    params.state.ddps[(uint32_t)DDP::YP], interval.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::ZP]);
    }
    else // Two dimensions
    {
      // Reset
      KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles,
                    params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::DYDTP],
                    params.state.ddps[(uint32_t)DDP::DRDTP], params.state.ddps[(uint32_t)DDP::ERROR],
                    params.state.ddps[(uint32_t)DDP::TEMP1], params.state.ddps[(uint32_t)DDP::TEMP2]);

      // Predict
      KERNEL_LAUNCH(predictKernel, kernelSize, 0, 0, params.state.numBubbles, timeStep,
                    params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::X],
                    params.state.ddps[(uint32_t)DDP::DXDT], params.state.ddps[(uint32_t)DDP::DXDTO],
                    params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::Y],
                    params.state.ddps[(uint32_t)DDP::DYDT], params.state.ddps[(uint32_t)DDP::DYDTO],
                    params.state.ddps[(uint32_t)DDP::RP], params.state.ddps[(uint32_t)DDP::R],
                    params.state.ddps[(uint32_t)DDP::DRDT], params.state.ddps[(uint32_t)DDP::DRDTO]);

      // Velocity
      KERNEL_LAUNCH(velocityPairKernel, params.state.pairKernelSize, 0, params.state.velocityStream,
                    params.env.getFZeroPerMuZero(), params.state.dips[(uint32_t)DIP::PAIR1],
                    params.state.dips[(uint32_t)DIP::PAIR2], params.state.ddps[(uint32_t)DDP::RP], interval.x, lbb.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::DXDTP],
                    interval.y, lbb.y, PBC_Y == 1, params.state.ddps[(uint32_t)DDP::YP],
                    params.state.ddps[(uint32_t)DDP::DYDTP]);
      // Wall velocity
      doWallVelocity(params.state.pairKernelSize, 0, params.state.velocityStream, PBC_X == 0, PBC_Y == 0, false,
                     params.state.numBubbles, params.state.dips[(uint32_t)DIP::PAIR1],
                     params.state.dips[(uint32_t)DIP::PAIR2], params.state.ddps[(uint32_t)DDP::RP],
                     params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                     params.state.ddps[(uint32_t)DDP::ZP], params.state.ddps[(uint32_t)DDP::DXDTP],
                     params.state.ddps[(uint32_t)DDP::DYDTP], params.state.ddps[(uint32_t)DDP::DZDTP], lbb, tfr,
                     properties);

      // Flow velocity
      if (USE_FLOW == 1)
      {
        CUDA_CALL(cudaMemset(params.state.dips[(uint32_t)DIP::TEMP1], 0, sizeof(int) * params.state.pairStride));
        int *numNeighbors = params.state.dips[(uint32_t)DIP::TEMP1];

        KERNEL_LAUNCH(neighborVelocityKernel, params.state.pairKernelSize, 0, params.state.velocityStream,
                      params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2], numNeighbors,
                      params.state.ddps[(uint32_t)DDP::TEMP1], params.state.ddps[(uint32_t)DDP::DXDTO],
                      params.state.ddps[(uint32_t)DDP::TEMP2], params.state.ddps[(uint32_t)DDP::DYDTO]);

        KERNEL_LAUNCH(flowVelocityKernel, params.state.pairKernelSize, 0, params.state.velocityStream,
                      params.state.numBubbles, numNeighbors, params.state.ddps[(uint32_t)DDP::DXDTP],
                      params.state.ddps[(uint32_t)DDP::DYDTP], params.state.ddps[(uint32_t)DDP::DZDTP],
                      params.state.ddps[(uint32_t)DDP::TEMP1], params.state.ddps[(uint32_t)DDP::TEMP2],
                      params.state.ddps[(uint32_t)DDP::TEMP3], params.state.ddps[(uint32_t)DDP::XP],
                      params.state.ddps[(uint32_t)DDP::YP], params.state.ddps[(uint32_t)DDP::ZP],
                      params.env.getFlowVel(), params.env.getFlowTfr(), params.env.getFlowLbb());
      }

      // Correct
      KERNEL_LAUNCH(correctKernel, kernelSize, 0, params.state.velocityStream, params.state.numBubbles, timeStep,
                    params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::XP],
                    params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::DXDT],
                    params.state.ddps[(uint32_t)DDP::DXDTP], params.state.ddps[(uint32_t)DDP::YP],
                    params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::DYDT],
                    params.state.ddps[(uint32_t)DDP::DYDTP]);

      // Path lenghts & distances
      KERNEL_LAUNCH(pathLengthDistanceKernel, kernelSize, 0, params.state.velocityStream, params.state.numBubbles,
                    params.state.ddps[(uint32_t)DDP::TEMP4], params.state.ddps[(uint32_t)DDP::PATH],
                    params.state.ddps[(uint32_t)DDP::DISTANCE], params.state.ddps[(uint32_t)DDP::XP],
                    params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::X0],
                    params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], interval.x, params.state.ddps[(uint32_t)DDP::YP],
                    params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::Y0],
                    params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP], interval.y);

      // Boundary wrap
      doBoundaryWrap(kernelSize, 0, params.state.velocityStream, PBC_X == 1, PBC_Y == 1, false, params.state.numBubbles,
                     params.state.ddps[(uint32_t)DDP::XP], params.state.ddps[(uint32_t)DDP::YP],
                     params.state.ddps[(uint32_t)DDP::ZP], lbb, tfr, params.state.dips[(uint32_t)DIP::WRAP_COUNT_X],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_Y], params.state.dips[(uint32_t)DIP::WRAP_COUNT_Z],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP], params.state.dips[(uint32_t)DIP::WRAP_COUNT_YP],
                     params.state.dips[(uint32_t)DIP::WRAP_COUNT_ZP]);

      // Gas exchange
      KERNEL_LAUNCH(gasExchangeKernel, params.state.pairKernelSize, 0, params.state.gasExchangeStream,
                    params.state.numBubbles, params.state.dips[(uint32_t)DIP::PAIR1],
                    params.state.dips[(uint32_t)DIP::PAIR2], params.state.ddps[(uint32_t)DDP::RP],
                    params.state.ddps[(uint32_t)DDP::DRDTP], params.state.ddps[(uint32_t)DDP::TEMP1], interval.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::XP], interval.y, PBC_Y == 1,
                    params.state.ddps[(uint32_t)DDP::YP]);
    }

    // Free area
    KERNEL_LAUNCH(freeAreaKernel, kernelSize, 0, params.state.gasExchangeStream, params.state.numBubbles,
                  params.state.ddps[(uint32_t)DDP::RP], params.state.ddps[(uint32_t)DDP::TEMP1],
                  params.state.ddps[(uint32_t)DDP::TEMP2], params.state.ddps[(uint32_t)DDP::TEMP3]);

    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP1],
                                                       params.state.dtfa, params.state.numBubbles,
                                                       params.state.gasExchangeStream);
    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP2],
                                                       params.state.dtfapr, params.state.numBubbles,
                                                       params.state.gasExchangeStream);
    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP3],
                                                       params.state.dta, params.state.numBubbles,
                                                       params.state.gasExchangeStream);

    KERNEL_LAUNCH(finalRadiusChangeRateKernel, kernelSize, 0, params.state.gasExchangeStream,
                  params.state.ddps[(uint32_t)DDP::DRDTP], params.state.ddps[(uint32_t)DDP::RP],
                  params.state.ddps[(uint32_t)DDP::TEMP1], params.state.numBubbles, params.env.getKappa(),
                  params.env.getKParameter());

    // Radius correct
    KERNEL_LAUNCH(correctKernel, kernelSize, 0, params.state.gasExchangeStream, params.state.numBubbles, timeStep,
                  params.state.ddps[(uint32_t)DDP::ERROR], params.state.ddps[(uint32_t)DDP::RP],
                  params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::DRDT],
                  params.state.ddps[(uint32_t)DDP::DRDTP]);

    // Calculate how many bubbles are below the minimum size.
    // Also take note of maximum radius.
    KERNEL_LAUNCH(setFlagIfGreaterThanConstantKernel, kernelSize, 0, params.state.gasExchangeStream,
                  params.state.numBubbles, params.state.dips[(uint32_t)DIP::FLAGS],
                  params.state.ddps[(uint32_t)DDP::RP], params.env.getMinRad());

    params.cw.reduceNoCopy<int, int *, int *>(&cub::DeviceReduce::Sum, params.state.dips[(uint32_t)DIP::FLAGS],
                                              static_cast<int *>(params.state.mbpc), params.state.numBubbles,
                                              params.state.gasExchangeStream);
    params.cw.reduceNoCopy<double, double *, double *>(&cub::DeviceReduce::Max, params.state.ddps[(uint32_t)DDP::RP],
                                                       static_cast<double *>(params.state.dtfa),
                                                       params.state.numBubbles, params.state.gasExchangeStream);

    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.state.pinnedInts), params.state.mbpc, sizeof(int),
                              cudaMemcpyDeviceToHost, params.state.gasExchangeStream));
    CUDA_CALL(cudaMemcpyAsync(static_cast<void *>(params.state.pinnedDoubles), params.state.dtfa, sizeof(double),
                              cudaMemcpyDeviceToHost, params.state.gasExchangeStream));

    // Error
    error = params.cw.reduce<double, double *, double *>(
      &cub::DeviceReduce::Max, params.state.ddps[(uint32_t)DDP::ERROR], params.state.numBubbles);

    if (error < params.env.getErrorTolerance() && timeStep < 0.1)
      timeStep *= 1.9;
    else if (error > params.env.getErrorTolerance())
      timeStep *= 0.5;

    ++numLoopsDone;

    NVTX_RANGE_POP();
  } while (error > params.env.getErrorTolerance());

  // Update values
  const size_t numBytesToCopy = 4 * sizeof(double) * params.state.dataStride;

  CUDA_CALL(cudaMemcpyAsync(params.state.ddps[(uint32_t)DDP::DXDTO], params.state.ddps[(uint32_t)DDP::DXDT],
                            numBytesToCopy, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(params.state.ddps[(uint32_t)DDP::X], params.state.ddps[(uint32_t)DDP::XP],
                            2 * numBytesToCopy, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(params.state.ddps[(uint32_t)DDP::PATH], params.state.ddps[(uint32_t)DDP::TEMP4],
                            sizeof(double) * params.state.dataStride, cudaMemcpyDeviceToDevice));
  CUDA_CALL(cudaMemcpyAsync(params.state.dips[(uint32_t)DIP::WRAP_COUNT_XP],
                            params.state.dips[(uint32_t)DIP::WRAP_COUNT_X], params.state.dataStride * 3 * sizeof(int),
                            cudaMemcpyDeviceToDevice));

  ++params.state.integrationStep;
  params.env.setTimeStep(timeStep);
  params.state.simulationTime += timeStep;

  params.state.maxBubbleRadius = params.state.pinnedDoubles[0];

  // Delete & reorder
  const int numBubblesAboveMinRad = params.state.pinnedInts[0];
  const bool shouldDeleteBubbles  = numBubblesAboveMinRad < params.state.numBubbles;

  if (shouldDeleteBubbles)
    deleteSmallBubbles(params, numBubblesAboveMinRad);

  if (shouldDeleteBubbles || params.state.integrationStep % 5000 == 0)
    updateCellsAndNeighbors(params);

  bool continueSimulation = params.state.numBubbles > params.env.getMinNumBubbles();
  continueSimulation &= (NUM_DIM == 3) ? params.state.maxBubbleRadius < 0.5 * (tfr - lbb).getMinComponent() : true;

  NVTX_RANGE_POP();

  return continueSimulation;
}

void transformPositions(Params &params, bool normalize)
{
  KERNEL_LAUNCH(transformPositionsKernel, params.state.pairKernelSize, 0, 0, normalize, params.state.numBubbles,
                params.env.getLbb(), params.env.getTfr(), params.state.ddps[(uint32_t)DDP::X],
                params.state.ddps[(uint32_t)DDP::Y], params.state.ddps[(uint32_t)DDP::Z]);
}

double calculateVolumeOfBubbles(Params &params)
{
  KernelSize kernelSize(128, params.state.numBubbles);

  KERNEL_LAUNCH(calculateVolumes, kernelSize, 0, 0, params.state.ddps[(uint32_t)DDP::R],
                params.state.ddps[(uint32_t)DDP::TEMP1], params.state.numBubbles);

  return params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP1],
                                                      params.state.numBubbles);
}

void deinit(Params &params)
{
  saveSnapshotToFile(params);
  params.env.writeParameters();

  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaFree(static_cast<void *>(params.state.deviceDoubles)));
  CUDA_CALL(cudaFree(static_cast<void *>(params.state.deviceInts)));
  CUDA_CALL(cudaFreeHost(static_cast<void *>(params.state.pinnedInts)));
  CUDA_CALL(cudaFreeHost(static_cast<void *>(params.state.pinnedDoubles)));

  CUDA_CALL(cudaStreamDestroy(params.state.velocityStream));
  CUDA_CALL(cudaStreamDestroy(params.state.gasExchangeStream));
}
} // namespace

namespace cubble
{
void run(const char *inputFileName, const char *outputFileName)
{
  Params params;
  params.env = Env(inputFileName, outputFileName);
  params.env.readParameters();

  std::cout << "\n=====\nSetup\n=====" << std::endl;
  {
    setup(params);
    saveSnapshotToFile(params);

    std::cout << "Letting bubbles settle after they've been created and before "
                 "scaling or stabilization."
              << std::endl;
    stabilize(params);
    saveSnapshotToFile(params);

    const double phiTarget = params.env.getPhiTarget();
    double bubbleVolume    = calculateVolumeOfBubbles(params);
    double phi             = bubbleVolume / params.env.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi << ", target: " << phiTarget << std::endl;

    std::cout << "Scaling the simulation box." << std::endl;
    transformPositions(params, true);
    dvec relativeSize = params.env.getBoxRelativeDimensions();
    relativeSize.z    = (NUM_DIM == 2) ? 1 : relativeSize.z;
    double t = calculateVolumeOfBubbles(params) / (phiTarget * relativeSize.x * relativeSize.y * relativeSize.z);
    t        = (NUM_DIM == 3) ? std::cbrt(t) : std::sqrt(t);
    params.env.setTfr(dvec(t, t, t) * relativeSize);
    transformPositions(params, false);

    phi = bubbleVolume / params.env.getSimulationBoxVolume();

    std::cout << "Volume ratios: current: " << phi << ", target: " << phiTarget << std::endl;

    saveSnapshotToFile(params);
  }

  std::cout << "\n=============\nStabilization\n=============" << std::endl;
  {
    int numSteps       = 0;
    const int failsafe = 500;

    std::cout << "#steps\tdE/t\te1\te2" << std::endl;
    while (true)
    {
      double time        = stabilize(params);
      double deltaEnergy = std::abs(params.state.energy2 - params.state.energy1) / time;
      deltaEnergy *= 0.5 * params.env.getSigmaZero();

      if (deltaEnergy < params.env.getMaxDeltaEnergy())
      {
        std::cout << "Final delta energy " << deltaEnergy << " after "
                  << (numSteps + 1) * params.env.getNumStepsToRelax() << " steps."
                  << " Energy before: " << params.state.energy1 << ", energy after: " << params.state.energy2
                  << ", time: " << time * params.env.getTimeScalingFactor() << std::endl;
        break;
      }
      else if (numSteps > failsafe)
      {
        std::cout << "Over " << failsafe * params.env.getNumStepsToRelax()
                  << " steps taken and required delta energy not reached."
                  << " Check parameters." << std::endl;
        break;
      }
      else
      {
        std::cout << (numSteps + 1) * params.env.getNumStepsToRelax() << "\t" << deltaEnergy << "\t"
                  << params.state.energy1 << "\t" << params.state.energy2 << std::endl;
      }

      ++numSteps;
    }

    saveSnapshotToFile(params);
  }

  std::cout << "\n==========\nSimulation\n==========" << std::endl;
  std::stringstream dataStream;
  {
    // Set starting positions and reset wrap counts to 0
    const size_t numBytesToCopy = 3 * sizeof(double) * params.state.dataStride;
    CUDA_CALL(cudaMemcpy(params.state.ddps[(uint32_t)DDP::X0], params.state.ddps[(uint32_t)DDP::X], numBytesToCopy,
                         cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemset(params.state.dips[(uint32_t)DIP::WRAP_COUNT_X], 0, 6 * params.state.dataStride * sizeof(int)));

    params.state.simulationTime = 0;
    int timesPrinted            = 1;
    uint32_t numSteps           = 0;
    const dvec interval         = params.env.getTfr() - params.env.getLbb();

    KernelSize kernelSize(128, params.state.numBubbles);

    // Calculate the energy at simulation start
    KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles, params.state.ddps[(uint32_t)DDP::TEMP4]);

    if (NUM_DIM == 3)
    {
      KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                    params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                    params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                    params.state.ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1, params.state.ddps[(uint32_t)DDP::Z]);
    }
    else
    {
      KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                    params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                    params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x,
                    PBC_X == 1, params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                    params.state.ddps[(uint32_t)DDP::Y]);
    }

    params.state.energy1 = params.cw.reduce<double, double *, double *>(
      &cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP4], params.state.numBubbles);

    // Start the simulation proper
    bool continueIntegration = true;
    int numTotalSteps        = 0;
    std::cout << "T\tphi\tR\t#b\tdE\t\t#steps\t#pairs" << std::endl;
    while (continueIntegration)
    {
      continueIntegration = integrate(params);
      CUDA_PROFILER_START(numTotalSteps == 2000);
      CUDA_PROFILER_STOP(numTotalSteps == 2200, continueIntegration);

      // The if clause contains many slow operations, but it's only done
      // very few times relative to the entire run time, so it should not
      // have a huge cost. Roughly 6e4-1e5 integration steps are taken for each
      // time step
      // and the if clause is executed once per time step.
      const double scaledTime = params.state.simulationTime * params.env.getTimeScalingFactor();
      if ((int)scaledTime >= timesPrinted)
      {
        kernelSize = KernelSize(128, params.state.numBubbles);
        // Calculate total energy
        KERNEL_LAUNCH(resetKernel, kernelSize, 0, 0, 0.0, params.state.numBubbles,
                      params.state.ddps[(uint32_t)DDP::TEMP4]);

        if (NUM_DIM == 3)
          KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                        params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                        params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x,
                        PBC_X == 1, params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                        params.state.ddps[(uint32_t)DDP::Y], interval.z, PBC_Z == 1,
                        params.state.ddps[(uint32_t)DDP::Z]);
        else
          KERNEL_LAUNCH(potentialEnergyKernel, params.state.pairKernelSize, 0, 0, params.state.numBubbles,
                        params.state.dips[(uint32_t)DIP::PAIR1], params.state.dips[(uint32_t)DIP::PAIR2],
                        params.state.ddps[(uint32_t)DDP::R], params.state.ddps[(uint32_t)DDP::TEMP4], interval.x,
                        PBC_X == 1, params.state.ddps[(uint32_t)DDP::X], interval.y, PBC_Y == 1,
                        params.state.ddps[(uint32_t)DDP::Y]);

        params.state.energy2 = params.cw.reduce<double, double *, double *>(
          &cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::TEMP4], params.state.numBubbles);
        const double dE = (params.state.energy2 - params.state.energy1) / params.state.energy2;

        // Add values to data stream
        const double averageRadius =
          params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::R],
                                                       params.state.numBubbles) /
          params.state.numBubbles;
        const double averagePath =
          params.cw.reduce<double, double *, double *>(&cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::PATH],
                                                       params.state.numBubbles) /
          params.state.numBubbles;
        const double averageDistance =
          params.cw.reduce<double, double *, double *>(
            &cub::DeviceReduce::Sum, params.state.ddps[(uint32_t)DDP::DISTANCE], params.state.numBubbles) /
          params.state.numBubbles;
        const double relativeRadius = averageRadius / params.env.getAvgRad();
        dataStream << scaledTime << " " << relativeRadius << " "
                   << params.state.maxBubbleRadius / params.env.getAvgRad() << " " << params.state.numBubbles << " "
                   << averagePath << " " << averageDistance << " " << dE << "\n";

        // Print some values
        std::cout << (int)scaledTime << "\t" << calculateVolumeOfBubbles(params) / params.env.getSimulationBoxVolume()
                  << "\t" << relativeRadius << "\t" << params.state.numBubbles << "\t" << dE << "\t" << numSteps << "\t"
                  << params.state.numPairs << std::endl;

        // Only write snapshots when t* is a power of 2.
        if ((timesPrinted & (timesPrinted - 1)) == 0)
          saveSnapshotToFile(params);

        ++timesPrinted;
        numSteps             = 0;
        params.state.energy1 = params.state.energy2;
      }

      ++numSteps;
      ++numTotalSteps;
    }
  }

  std::ofstream file(params.env.getDataFilename());
  file << dataStream.str() << std::endl;

  deinit(params);
}
} // namespace cubble
