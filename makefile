CC = gcc
CC_FLAGS = -w -Wall -O0 -g
LD_FLAGS = -lm

CU = nvcc
CU_FLAGS = -O2 -diag-suppress 2464 -diag-suppress 1650 #-g -G #-Xptxas 

all:
	$(CU) $(CU_FLAGS) cnn.cu main.cu lenet.cpp -o cnn-par
	$(CC) $(CC_FLAGS) lenet/lenet.c lenet/main.c -o cnn-seq $(LD_FLAGS)