CC = gcc
CC_FLAGS = -w -Wall -O2
LD_FLAGS = -lm

CU = nvcc
CU_FLAGS = -Xptxas -O2 -diag-suppress 2464

all:
	$(CU) $(CU_FLAGS) cnn.cu -o cnn-par
	$(CC) $(CC_FLAGS) lenet/lenet.c lenet/main.c -o cnn-seq $(LD_FLAGS)