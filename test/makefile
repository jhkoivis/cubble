# -----------------------------------------------------
# How many dimensions to simulate.
# -----------------------------------------------------

NUM_DIM := 3


# -----------------------------------------------------
# Which dimensions use periodic boundary conditions?
# -----------------------------------------------------

PBC_X := 1
PBC_Y := 1
PBC_Z := 1


# -----------------------------------------------------
# Use flow or not?
# -----------------------------------------------------

USE_FLOW := 0


# -----------------------------------------------------
# Profile or not?
# -----------------------------------------------------

USE_PROFILING := 0


# -----------------------------------------------------
# Directories
# -----------------------------------------------------

# bin will be created when building the program.
# Executable and the intermediate object files will be placed there.
# 'make clean' will completely remove bin and its contents.
BIN_PATH := bin

# All the source files recide here.
SRC_PATH := ../src

# Included (external) headers
INCL = -I../incl/ -I./ -I../incl/googletest/include/


# -----------------------------------------------------
# Object files, headers and the main executable
# -----------------------------------------------------

# List all objects that contain CPU code.
OBJ_NAMES := gtest_main.o Kernels.o lol_cuda.o
OBJS := $(addprefix $(BIN_PATH)/, $(OBJ_NAMES))

# List all the objects that contain GPU code.
# Overlap with the objects above is totally fine.
# These are only related to linking, compiling is done automatically
# based on the file extension (.cpp vs. .cu)
GPU_OBJ_NAMES := Kernels.o lol_cuda.o
GPU_OBJS := $(addprefix $(BIN_PATH)/, $(GPU_OBJ_NAMES))

# Find all headers in source dir.
HEADERS := $(wildcard $(SRC_PATH)/*.h)
HEADERS += $(wildcard $(SRC_PATH)/*.cuh)

# Name of the linked GPU code.
GPU_CODE := $(BIN_PATH)/GPUCode.o

# Name of the final executable.
EXEC := $(BIN_PATH)/test_cubble

# -----------------------------------------------------
# Compilers to use
# -----------------------------------------------------

C_CPU := g++
C_GPU := nvcc


# -----------------------------------------------------
# External libraries to link
# -----------------------------------------------------

LIBS := -lcudart -lcurand -lnvToolsExt -L../incl/googletest/lib64 -lgtest -lgmock -lgtest_main


# -----------------------------------------------------
# Preprocessor defines
# -----------------------------------------------------

DEFINES := -DNUM_DIM=$(NUM_DIM) -DUSE_FLOW=$(USE_FLOW) \
	   -DPBC_X=$(PBC_X) -DPBC_Y=$(PBC_Y) -DPBC_Z=$(PBC_Z) \
	   -DNDEBUG -DUSE_PROFILING=$(USE_PROFILING)
NVCC_DEFINES := -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__


# -----------------------------------------------------
# Flags
# -----------------------------------------------------

CPU_FLAGS := -Wall -std=c++14 -m64 -pthread -g
GPU_ARCH := -gencode=arch=compute_60,code=sm_60 \
	    -gencode=arch=compute_70,code=sm_70
GPU_FLAGS := $(GPU_ARCH) -std=c++11 --expt-relaxed-constexpr -lineinfo
COMMON_FLAGS := $(INCL)
OPTIM_FLAGS := -O3
PROFILING_FLAGS :=
ifeq ($(USE_PROFILING), 1)
PROFILING_FLAGS := -pg
endif


# -----------------------------------------------------
# First rule: Builds the project with default settings
# -----------------------------------------------------

.PHONY : all
all : | $(BIN_PATH)
	$(MAKE) -j8 $(EXEC)


# -----------------------------------------------------
# Rule for main executable.
# -----------------------------------------------------

# By default has some safety nets at place but also uses some optimizations.
$(EXEC) : $(HEADERS) $(OBJS) $(GPU_CODE)
	$(C_CPU) $(CPU_FLAGS) $(PROFILING_FLAGS) $(OBJS) $(GPU_CODE) $(LIBS) -o $@ $(INCL)


# -----------------------------------------------------
# Rule for linking the GPU code to a single object file
# -----------------------------------------------------

$(GPU_CODE) : $(GPU_OBJS)
	$(C_GPU) $(PROFILING_FLAGS) $(GPU_ARCH) -dlink $^ -o $@


# -----------------------------------------------------
# Rule for the intermediate objects
# -----------------------------------------------------

# CPU code
$(BIN_PATH)/%.o : $(SRC_PATH)/%.cpp
	$(eval OPTIONS = $(CPU_FLAGS) $(COMMON_FLAGS) \
	$(OPTIM_FLAGS) $(DEFINES) $(PROFILING_FLAGS))
	$(C_CPU) $(CPU_FLAGS) $< $(OPTIONS) -c $(INCL) -o $@

# GPU code
$(BIN_PATH)/%.o : $(SRC_PATH)/%.cu
	$(eval OPTIONS = $(GPU_FLAGS) $(COMMON_FLAGS) \
	$(OPTIM_FLAGS) $(DEFINES) $(NVCC_DEFINES) $(PROFILING_FLAGS))
	$(C_GPU) $< $(OPTIONS) -dc $(INCL) -o $@

# -----------------------------------------------------
# Clean up
# -----------------------------------------------------

.PHONY : clean
clean :
	rm -rf $(BIN_PATH)


# -----------------------------------------------------
# mkdir bin
# -----------------------------------------------------

$(BIN_PATH) :
	mkdir $(BIN_PATH)
