#include "Kernels.cuh"

namespace cubble
{
__constant__ __device__ double dTotalArea;
__constant__ __device__ double dTotalFreeArea;
__constant__ __device__ double dTotalFreeAreaPerRadius;
__constant__ __device__ double dTotalVolume;
__device__ bool dErrorEncountered;
__device__ int dNumPairs;
__device__ double dVolumeMultiplier;
__device__ double dInvRho;

__device__ void logError(bool condition, const char *statement,
                         const char *errMsg)
{
  if (condition == false)
  {
    printf("----------------------------------------------------"
           "\nError encountered"
           "\n(%s) -> %s"
           "\n@thread[%d, %d, %d], @block[%d, %d, %d]"
           "\n----------------------------------------------------\n",
           statement, errMsg, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x,
           blockIdx.y, blockIdx.z);

    dErrorEncountered = true;
  }
}

__device__ int getGlobalTid()
{
  // Simple helper function for calculating a 1D coordinate
  // from 1, 2 or 3 dimensional coordinates.
  int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
  int blocksBefore =
    blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.y * gridDim.x + blockIdx.x;
  int threadsBefore =
    blockDim.y * blockDim.x * threadIdx.z + blockDim.x * threadIdx.y;
  int tid = blocksBefore * threadsPerBlock + threadsBefore + threadIdx.x;

  return tid;
}
__device__ void resetDoubleArrayToValue(double value, int idx, double *array)
{
  array[idx] = value;
}

__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values,
                                          double constant)
{
  flags[idx] = values[idx] < constant ? 1 : 0;
}

__device__ void setFlagIfGreaterThanConstant(int idx, int *flags,
                                             double *values, double constant)
{
  flags[idx] = values[idx] > constant ? 1 : 0;
}

__device__ double getWrappedDistance(double x1, double x2, double maxDistance,
                                     bool shouldWrap)
{
  const double distance = x1 - x2;
  x2                    = distance < -0.5 * maxDistance
         ? x2 - maxDistance
         : (distance > 0.5 * maxDistance ? x2 + maxDistance : x2);
  const double distance2 = x1 - x2;

  return shouldWrap ? distance2 : distance;
}

__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance,
                                     bool shouldWrap, double *x)
{
  const double distance =
    getWrappedDistance(x[idx1], x[idx2], maxDistance, shouldWrap);
  DEVICE_ASSERT(distance * distance > 0, "Distance is zero!");
  return distance * distance;
}
__device__ double getDistanceSquared(int idx1, int idx2, double maxDistance,
                                     double minDistance, bool shouldWrap,
                                     double *x, double *useless)
{
  return getDistanceSquared(idx1, idx2, maxDistance, shouldWrap, x);
}

__global__ void transformPositionsKernel(bool normalize, int numValues,
                                         dvec lbb, dvec tfr, double *x,
                                         double *y, double *z)
{
  const dvec interval = tfr - lbb;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    if (normalize)
    {
      x[i] = (x[i] - lbb.x) / interval.x;
      y[i] = (y[i] - lbb.y) / interval.y;
#if (NUM_DIM == 3)
      z[i] = (z[i] - lbb.z) / interval.z;
#endif
    }
    else
    {
      x[i] = interval.x * x[i] + lbb.x;
      y[i] = interval.y * y[i] + lbb.y;
#if (NUM_DIM == 3)
      z[i] = interval.z * z[i] + lbb.z;
#endif
    }
  }
}

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum)
{
  ivec idxVec = cellIdx;
  switch (neighborNum)
  {
    case 0:
      // self
      break;
    case 1:
      idxVec += ivec(-1, 1, 0);
      break;
    case 2:
      idxVec += ivec(-1, 0, 0);
      break;
    case 3:
      idxVec += ivec(-1, -1, 0);
      break;
    case 4:
      idxVec += ivec(0, -1, 0);
      break;
    case 5:
      idxVec += ivec(-1, 1, -1);
      break;
    case 6:
      idxVec += ivec(-1, 0, -1);
      break;
    case 7:
      idxVec += ivec(-1, -1, -1);
      break;
    case 8:
      idxVec += ivec(0, 1, -1);
      break;
    case 9:
      idxVec += ivec(0, 0, -1);
      break;
    case 10:
      idxVec += ivec(0, -1, -1);
      break;
    case 11:
      idxVec += ivec(1, 1, -1);
      break;
    case 12:
      idxVec += ivec(1, 0, -1);
      break;
    case 13:
      idxVec += ivec(1, -1, -1);
      break;
    default:
      printf("Should never end up here!\n");
      break;
  }

#if (PBC_X == 1)
  idxVec.x += dim.x;
  idxVec.x %= dim.x;
#else
  if (idxVec.x < 0 || idxVec.x >= dim.x)
    return -1;
#endif

#if (PBC_Y == 1)
  idxVec.y += dim.y;
  idxVec.y %= dim.y;
#else
  if (idxVec.y < 0 || idxVec.y >= dim.y)
    return -1;
#endif

#if (PBC_Z == 1)
  idxVec.z += dim.z;
  idxVec.z %= dim.z;
#else
  if (idxVec.z < 0 || idxVec.z >= dim.z)
    return -1;
#endif

  return get1DIdxFrom3DIdx(idxVec, dim);
}

__device__ double getWrappedCoordinate(double val1, double val2,
                                       double multiplier)
{
  double difference = val1 - val2;
  val2              = difference < -0.5 * multiplier
           ? val2 - multiplier
           : (difference > 0.5 * multiplier ? val2 + multiplier : val2);
  val2 = val1 - val2;

  return val2;
}

__device__ int getCellIdxFromPos(double x, double y, double z, dvec lbb,
                                 dvec tfr, ivec cellDim)
{
  const dvec interval = tfr - lbb;
  const int xid       = floor(cellDim.x * (x - lbb.x) / interval.x);
  const int yid       = floor(cellDim.y * (y - lbb.y) / interval.y);
#if (NUM_DIM == 3)
  const int zid = floor(cellDim.z * (z - lbb.z) / interval.z);
#else
  const int zid = 0;
#endif

  return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}

__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim)
{
// Linear encoding
// return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;

// Morton encoding
#if (NUM_DIM == 3)
  return encodeMorton3((unsigned int)idxVec.x, (unsigned int)idxVec.y,
                       (unsigned int)idxVec.z);
#else
  return encodeMorton2((unsigned int)idxVec.x, (unsigned int)idxVec.y);
#endif
}

__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim)
{
  ivec idxVec(0, 0, 0);
// Linear decoding
/*
idxVec.x = idx % cellDim.x;
idxVec.y = (idx / cellDim.x) % cellDim.y;
#if (NUM_DIM == 3)
idxVec.z = idx / (cellDim.x * cellDim.y);
#endif
*/
#if (NUM_DIM == 3)
  idxVec.x = decodeMorton3x((unsigned int)idx);
  idxVec.y = decodeMorton3y((unsigned int)idx);
  idxVec.z = decodeMorton3z((unsigned int)idx);
#else
  idxVec.x     = decodeMorton2x((unsigned int)idx);
  idxVec.y     = decodeMorton2y((unsigned int)idx);
#endif

  return idxVec;
}

__device__ __host__ unsigned int encodeMorton2(unsigned int x, unsigned int y)
{
  return (part1By1(y) << 1) + part1By1(x);
}

__device__ __host__ unsigned int encodeMorton3(unsigned int x, unsigned int y,
                                               unsigned int z)
{
  return (part1By2(z) << 2) + (part1By2(y) << 1) + part1By2(x);
}

__device__ __host__ unsigned int decodeMorton2x(unsigned int code)
{
  return compact1By1(code >> 0);
}

__device__ __host__ unsigned int decodeMorton2y(unsigned int code)
{
  return compact1By1(code >> 1);
}

__device__ __host__ unsigned int decodeMorton3x(unsigned int code)
{
  return compact1By2(code >> 0);
}

__device__ __host__ unsigned int decodeMorton3y(unsigned int code)
{
  return compact1By2(code >> 1);
}

__device__ __host__ unsigned int decodeMorton3z(unsigned int code)
{
  return compact1By2(code >> 2);
}

__device__ __host__ unsigned int part1By1(unsigned int x)
{
  // Mask the lowest 16 bits
  x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x =
    (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x =
    (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x =
    (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x =
    (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

  return x;
}

__device__ __host__ unsigned int part1By2(unsigned int x)
{
  // Mask lowest 10 bits
  x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
  x =
    (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x =
    (x ^ (x << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x =
    (x ^ (x << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x =
    (x ^ (x << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

  return x;
}

__device__ __host__ unsigned int compact1By1(unsigned int x)
{
  x &= 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x =
    (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
  x =
    (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x =
    (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x =
    (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
  return x;
}

__device__ __host__ unsigned int compact1By2(unsigned int x)
{
  x &= 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x =
    (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  x =
    (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x =
    (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x =
    (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210

  return x;
}

__device__ void wrapAround(int idx, double *coordinate, double minValue,
                           double maxValue, int *wrapMultiplier,
                           int *wrapMultiplierPrev)
{
  const double interval = maxValue - minValue;
  double value          = coordinate[idx];
  int multiplier        = wrapMultiplierPrev[idx];

  const bool smaller = value < minValue;
  const bool larger  = value > maxValue;

  value = smaller ? value + interval : (larger ? value - interval : value);
  multiplier =
    smaller ? multiplier - 1 : (larger ? multiplier + 1 : multiplier);

  wrapMultiplier[idx] = multiplier;
  coordinate[idx]     = value;
}

__device__ void addVelocity(int idx1, int idx2, double multiplier,
                            double maxDistance, double minDistance,
                            bool shouldWrap, double *x, double *v)
{
  const double velocity =
    getWrappedDistance(x[idx1], x[idx2], maxDistance, shouldWrap) * multiplier;
  atomicAdd(&v[idx1], velocity);
  atomicAdd(&v[idx2], -velocity);
}


__device__ void addNeighborVelocity(int idx1, int idx2, double *sumOfVelocities,
                                    double *velocity)
{
  atomicAdd(&sumOfVelocities[idx1], velocity[idx2]);
  atomicAdd(&sumOfVelocities[idx2], velocity[idx1]);
}

__global__ void calculateVolumes(double *r, double *volumes, int numValues)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    double radius = r[i];
    double volume = radius * radius * CUBBLE_PI;
#if (NUM_DIM == 3)
    volume *= radius * 1.33333333333333333333333333;
#endif

    volumes[i] = volume;
  }
}

__global__ void assignDataToBubbles(double *x, double *y, double *z,
                                    double *xPrd, double *yPrd, double *zPrd,
                                    double *r, double *w, int *aboveMinRadFlags,
                                    int *indices, ivec bubblesPerDim, dvec tfr,
                                    dvec lbb, double avgRad, double minRad,
                                    int numValues)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    indices[i] = i;
    dvec pos(0, 0, 0);
    pos.x = (i % bubblesPerDim.x) / (double)bubblesPerDim.x;
    pos.y = ((i / bubblesPerDim.x) % bubblesPerDim.y) / (double)bubblesPerDim.y;

    dvec randomOffset(x[i], y[i], 0);
#if (NUM_DIM == 3)
    randomOffset.z = z[i];
    pos.z = (i / (bubblesPerDim.x * bubblesPerDim.y)) / (double)bubblesPerDim.z;
#endif
    dvec interval = tfr - lbb;
    pos *= interval;
    randomOffset = dvec::normalize(randomOffset) * avgRad * w[i];
    pos += randomOffset;

    r[i] = r[i] > 0 ? r[i] : -r[i];

    x[i] = pos.x > lbb.x ? (pos.x < tfr.x ? pos.x : pos.x - interval.x)
                         : pos.x + interval.x;
    y[i] = pos.y > lbb.y ? (pos.y < tfr.y ? pos.y : pos.y - interval.y)
                         : pos.y + interval.y;
    z[i] = pos.z > lbb.z ? (pos.z < tfr.z ? pos.z : pos.z - interval.z)
                         : pos.z + interval.z;

    xPrd[i] = pos.x;
    yPrd[i] = pos.y;
    zPrd[i] = pos.z;

    w[i] = 2.0 * CUBBLE_PI * r[i] / numValues;

#if (NUM_DIM == 3)
    w[i] *= 2.0 * r[i];
#endif

    setFlagIfGreaterThanConstant(i, aboveMinRadFlags, r, minRad);
  }
}

__global__ void assignBubblesToCells(double *x, double *y, double *z,
                                     int *cellIndices, int *bubbleIndices,
                                     dvec lbb, dvec tfr, ivec cellDim,
                                     int numValues)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    cellIndices[i]   = getCellIdxFromPos(x[i], y[i], z[i], lbb, tfr, cellDim);
    bubbleIndices[i] = i;
  }
}

__global__ void velocityWallKernel(int numValues, double *r, double *x,
                                   double *y, double *z, double *vx, double *vy,
                                   double *vz, dvec lbb, dvec tfr,
                                   double fZeroPerMuZero, double dragCoeff)
{
#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
  const int tid = getGlobalTid();
  if (tid < numValues)
  {
    double distance1 = 0.0;
    double distance2 = 0.0;
    double distance  = 0.0;
    double xDrag     = 1.0;
    double yDrag     = 1.0;

#if (PBC_X == 0)
    distance1 = x[tid] - lbb.x;
    distance2 = x[tid] - tfr.x;
    distance =
      distance1 * distance1 < distance2 * distance2 ? distance1 : distance2;
    if (r[tid] * r[tid] >= distance * distance)
    {
      const double direction = distance < 0 ? -1.0 : 1.0;
      distance *= direction;
      const double velocity = direction * distance * fZeroPerMuZero *
                              (r[tid] - distance) / (r[tid] * distance);
      vx[tid] += velocity;
      xDrag = 1.0 - dragCoeff;

      // Drag of x wall to y & z
      vy[tid] *= xDrag;
      vz[tid] *= xDrag;
    }
#endif

#if (PBC_Y == 0)
    distance1 = y[tid] - lbb.y;
    distance2 = y[tid] - tfr.y;
    distance =
      distance1 * distance1 < distance2 * distance2 ? distance1 : distance2;
    if (r[tid] * r[tid] >= distance * distance)
    {
      const double direction = distance < 0 ? -1.0 : 1.0;
      distance *= direction;
      const double velocity = direction * distance * fZeroPerMuZero *
                              (r[tid] - distance) / (r[tid] * distance);

      // Retroactively apply possible drag from x wall to the velocity the y
      // wall causes
      vy[tid] += velocity * xDrag;
      yDrag = 1.0 - dragCoeff;

      // Drag of y wall to x & z
      vx[tid] *= yDrag;
      vz[tid] *= yDrag;
    }
#endif

#if (PBC_Z == 0)
    distance1 = z[tid] - lbb.z;
    distance2 = z[tid] - tfr.z;
    distance =
      distance1 * distance1 < distance2 * distance2 ? distance1 : distance2;
    if (r[tid] * r[tid] >= distance * distance)
    {
      const double direction = distance < 0 ? -1.0 : 1.0;
      distance *= direction;
      const double velocity = direction * distance * fZeroPerMuZero *
                              (r[tid] - distance) / (r[tid] * distance);

      // Retroactively apply possible drag from x & y walls to the velocity the
      // z wall causes
      vz[tid] += velocity * xDrag * yDrag;

      // Drag of z wall to x & y directions
      vx[tid] *= 1.0 - dragCoeff;
      vy[tid] *= 1.0 - dragCoeff;
    }
#endif
  }
#else
  return;
#endif
}

__global__ void flowVelocityKernel(int numValues, int *numNeighbors,
                                   double *velX, double *velY, double *velZ,
                                   double *nVelX, double *nVelY, double *nVelZ,
                                   double *posX, double *posY, double *posZ,
                                   double *r, dvec flowVel, dvec flowTfr,
                                   dvec flowLbb)
{
  /* Function that implements a flow region inside the simulation box.
   *
   * Loops through each bubble. Bubble should be considered to be inside the
   * flow region, if its centre (posX, posY, posZ) is within the the flow box or
   * if the bubble is touching or overlapping the boundary of the box due to its
   * spacial extent, i.e. the bubble is large enough to partially cross into the
   * flow region. If the bubble is found to be inside the flow region its
   * velocity is adjusted by the flow velocity
   */
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    const double multiplier =
      (numNeighbors[i] > 0 ? 1.0 / numNeighbors[i] : 0.0);

    // Checks if the bubble centre point is within the flow region
    // OR the distance between the lowest boundary and the centre is smaller or
    // equal to the radius OR the distance between the highest boundary and the
    // centre is smaller or equal to the radius If any condition True then
    // inside will be 1, else 0
    int inside =
      (int)((posX[i] < flowTfr.x && posX[i] > flowLbb.x) ||
            ((flowLbb.x - posX[i]) * (flowLbb.x - posX[i]) <= r[i] * r[i]) ||
            ((flowTfr.x - posX[i]) * (flowTfr.x - posX[i]) <= r[i] * r[i]));

    // Repeat same for y. Multiply result, since conditions must be satisfied in
    // all coordinate directions
    inside *=
      (int)((posY[i] < flowTfr.y && posY[i] > flowLbb.y) ||
            ((flowLbb.y - posY[i]) * (flowLbb.y - posY[i]) <= r[i] * r[i]) ||
            ((flowTfr.y - posY[i]) * (flowTfr.y - posY[i]) <= r[i] * r[i]));

#if (NUM_DIM == 3)
    inside *=
      (int)((posZ[i] < flowTfr.z && posZ[i] > flowLbb.z) ||
            ((flowLbb.z - posZ[i]) * (flowLbb.z - posZ[i]) <= r[i] * r[i]) ||
            ((flowTfr.z - posZ[i]) * (flowTfr.z - posZ[i]) <= r[i] * r[i]));

    velZ[i] += !inside * multiplier * nVelZ[i] + flowVel.z * inside;
#endif

    // Velocities of the bubble will only change if inside is 1, i.e. it
    // fulfilled the conditions in all coordinate directions
    velX[i] += !inside * multiplier * nVelX[i] + flowVel.x * inside;
    velY[i] += !inside * multiplier * nVelY[i] + flowVel.y * inside;
  }
}

__global__ void freeAreaKernel(int numValues, double *r, double *freeArea,
                               double *freeAreaPerRadius, double *area)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    double totalArea = 2.0 * CUBBLE_PI * r[i];
#if (NUM_DIM == 3)
    totalArea *= 2.0 * r[i];
#endif
    area[i]              = totalArea;
    freeArea[i]          = totalArea - freeArea[i];
    freeAreaPerRadius[i] = freeArea[i] / r[i];
  }
}

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r,
                                            double *freeArea, int numValues,
                                            double kappa, double kParam,
                                            double averageSurfaceAreaIn)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    dInvRho                = dTotalFreeAreaPerRadius / dTotalFreeArea;
    const double invRadius = 1.0 / r[i];
    double invArea         = 0.5 * CUBBLE_I_PI * invRadius;
#if (NUM_DIM == 3)
    invArea *= 0.5 * invRadius;
#endif
    const double vr = drdt[i] +
                      kappa * averageSurfaceAreaIn * numValues / dTotalArea *
                        freeArea[i] * (dInvRho - invRadius);
    drdt[i] = kParam * invArea * vr;
  }
}

__global__ void addVolume(double *r, int numValues)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    double multiplier = dVolumeMultiplier / dTotalVolume;
    multiplier += 1.0;

#if (NUM_DIM == 3)
    multiplier = cbrt(multiplier);
#else
    multiplier = sqrt(multiplier);
#endif
    r[i] = r[i] * multiplier;
  }
}

__global__ void calculateRedistributedGasVolume(double *volume, double *r,
                                                int *aboveMinRadFlags,
                                                int numValues)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
       i += gridDim.x * blockDim.x)
  {
    const double radius = r[i];
    double vol          = CUBBLE_PI * radius * radius;
#if (NUM_DIM == 3)
    vol *= 1.333333333333333333333333 * radius;
#endif

    if (aboveMinRadFlags[i] == 0)
    {
      atomicAdd(&dVolumeMultiplier, vol);
      volume[i] = 0;
    }
    else
      volume[i] = vol;
  }
}

__device__ void adamsBashforth(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fPrevious)
{
  yNext[idx] = y[idx] + 0.5 * timeStep * (3.0 * f[idx] - fPrevious[idx]);
}

__device__ double adamsMoulton(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fNext)
{
  const double corrected = y[idx] + 0.5 * timeStep * (f[idx] + fNext[idx]);
  const double error     = corrected - yNext[idx];
  yNext[idx]             = corrected;

  return error < 0 ? -error : error;
}

__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f)
{
  y[idx] += f[idx] * timeStep;
}

__device__ double calculateDistanceFromStart(int idx, double *x, double *xPrev,
                                             double *xStart,
                                             int *wrapMultiplier,
                                             double interval)
{
  double distance = x[idx] - xStart[idx] + wrapMultiplier[idx] * interval;
  return distance * distance;
}

__device__ double calculatePathLength(int idx, double *x, double *xPrev,
                                      double *xStart, int *wrapMultiplier,
                                      double interval)
{
  // Only works if done before boundary wrap
  const double diff = x[idx] - xPrev[idx];
  return diff * diff;
}

int test_me(void){
	return 27;
}

__device__ void ctst(float *x, float *y, float *z, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	printf("thread id = %i\n", i);
	z[i] = x[i] + y[i];
}

} // namespace cubble

