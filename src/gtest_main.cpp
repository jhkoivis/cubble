//
// Created by Evgeni Ulanov on 7/16/19.
// Copyright (c) 2019
//
#include "gtest/gtest.h"
#include "lol_cuda.cuh"

TEST(test_timesFive, integerTests){
simple(5);

TestSuite::simple_cuda () ;
	EXPECT_EQ(5, 5);
}

int main(int argc, char* argv[]){
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}


