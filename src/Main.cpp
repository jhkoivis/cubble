#include "Util.h"
#include <cuda.h>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>

namespace cubble
{
void run(std::string &&inputFileName, std::string &&outputFileName,
         int localRank, int localSize, const MPI_Comm &shared);
}

int main(int argc, char **argv)
{
  std::exception_ptr pExc = nullptr;

  if (argc != 3)
  {
    std::cout << "\nUsage: " << argv[0] << " inputFile outputFile"
              << "\ninputFile = the name of the (.json) file that contains"
              << " the necessary inputs, or the name of the binary file that "
                 "contains the serialized state of a non-finished "
                 "simulation.\noutputFile = (.bin) file name where to save "
                 "data if simulation ends before completion"
              << std::endl;

    return EXIT_FAILURE;
  }

  int returnCode = -1337;

  try
  {
    std::cout << "-------------------------------------------------------------"
                 "-----------\n"
              << "The current program simulates the bubbles in " << NUM_DIM
              << " dimensions.\n"
              << "If you want to change the dimensionality of the program, "
                 "change the number of dimensions 'NUM_DIM'"
              << "\nin Makefile and rebuild the program.\n"
              << "-------------------------------------------------------------"
                 "-----------\n"
              << std::endl;

    MPI_Comm shared = NULL;
    int localRank   = -1337;
    int localSize   = -1337;
    int numGpus     = -1337;

    returnCode = MPI_Init(&argc, &argv);

    if (returnCode != MPI_SUCCESS)
    {
      std::cout << "MPI Initialization failed!\n" << returnCode << std::endl;
      return EXIT_FAILURE;
    }

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shared);
    MPI_Comm_size(shared, &localSize); // number of ranks in this node
    MPI_Comm_rank(shared, &localRank); // my local rank
    CUDA_CALL(cudaGetDeviceCount(&numGpus));

    if (numGpus == localSize)
      CUDA_CALL(cudaSetDevice(localRank));
    else
    {
      std::cout
        << "Local size (mpi) different from number of devices (cuda gpu)"
        << std::endl;
      return EXIT_FAILURE;
    }

    cubble::run(std::string(argv[1]), std::string(argv[2]), localRank,
                localSize, shared);
  }
  catch (const std::exception &e)
  {
    pExc = std::current_exception();
    cubble::handleException(pExc);

    return EXIT_FAILURE;
  }

  cudaDeviceReset();
  returnCode = MPI_Finalize();

  if (returnCode != MPI_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
