#include "gtest/gtest.h"
#include "cubble_tests.cuh"

int main(int argc, char* argv[]){
	testing::InitGoogleTest(&argc, argv);  // Finds all TEST cases
	return RUN_ALL_TESTS();  // Runs all tests
}


