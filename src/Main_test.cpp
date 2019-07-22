#include "Util.h"
#include <cuda.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tests.cuh"

using namespace std;
using namespace TestSuite;

int main(int argc, char **argv)
{
  std::cout << "run" << std::endl;
  TEST_tests_are_running();
  TEST_cubble_simple();
  TEST_simple_kernel();
}
