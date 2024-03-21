#include "cnn.h"
#include "lenet.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE  "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL  "t10k-labels-idx1-ubyte"

#define COUNT_TRAIN 60000
#define COUNT_TEST  10000

#define IMAGE_WIDTH 28

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))



//~ Initializaiton functions

void NetworkCreateLeNet5Reference(Network* network)
{
	const int kernelWidth = 5;
    
	NetworkCreate(network, 7);
    
	// Add the layers
	NetworkAddLayerInput2D(network, 1, 28, 28, 4);
	NetworkAddLayerConvolution2D(network, 6, kernelWidth, ActivationTypeReLU);
	NetworkAddLayerPooling2D(network, 2, PoolingTypeMax);
	NetworkAddLayerConvolution2D(network, 16, kernelWidth, ActivationTypeReLU);
	NetworkAddLayerPooling2D(network, 2, PoolingTypeMax);
	NetworkAddLayerConvolution2D(network, 120, kernelWidth, ActivationTypeReLU);
	NetworkAddLayerFullyConnected(network, 10, ActivationTypeReLU);
}

void NetworkCreateLeNet5Real(Network* network)
{
	const int kernelWidth = 5;
    
	NetworkCreate(network, 8);
    
	// Add the layers
	NetworkAddLayerInput2D(network, 1, 28, 28, 4);
	NetworkAddLayerConvolution2D(network, 6, kernelWidth, ActivationTypeReLU);
	NetworkAddLayerPooling2D(network, 2, PoolingTypeMax);
	NetworkAddLayerConvolution2D(network, 16, kernelWidth, ActivationTypeReLU);
	NetworkAddLayerPooling2D(network, 2, PoolingTypeMax);
	NetworkAddLayerFullyConnected(network, 120, ActivationTypeReLU);
	NetworkAddLayerFullyConnected(network, 84, ActivationTypeReLU);
	NetworkAddLayerFullyConnected(network, 10, ActivationTypeReLU);
}

void NetworkCreateSimple(Network* network)
{
	NetworkCreate(network, 5);
    
	// Add the layers
	NetworkAddLayerInput2D(network, 1, 28, 28, 0);
	NetworkAddLayerFullyConnected(network, 256, ActivationTypeReLU);
	NetworkAddLayerFullyConnected(network, 128, ActivationTypeReLU);
	NetworkAddLayerFullyConnected(network, 10, ActivationTypeReLU);
}

void NetworkCreateNNADL(Network* network)
{
	NetworkCreate(network, 4);
    
	// Add the layers
	NetworkAddLayerInput2D(network, 1, 28, 28, 0);
	NetworkAddLayerFullyConnected(network, 30, ActivationTypeReLU);
	NetworkAddLayerFullyConnected(network, 10, ActivationTypeReLU);
}



//~ Debugging

void RunStock(Dataset datasetTrain, Dataset datasetTest)
{
	// Initialized LeNet
	LeNet5* lenet = (LeNet5*)malloc(sizeof(LeNet5));
	Initial(lenet);
    
	int batches[] = { 300 };
	// We are using only one epoch, even though multiple epochs have their benefits.
	for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
	{
		for (int j = 0; j <= datasetTrain.count - batches[i]; j += batches[i])
		{
			TrainBatch(lenet, (image*)datasetTrain.images + i, (uint8*)datasetTrain.labels + i, batches[i]);
		}
	}
    
	int confusion_matrix[10][10] = { 0 }; // For our specific problem, we have a 10x10 confusion matrix 
	int right = 0;
	for (int i = 0; i < datasetTest.count; ++i)
	{
		uint8 l = (uint8)datasetTest.labels[i];
		int p = Predict(lenet, ((image*)datasetTest.images)[i], 10);
		confusion_matrix[l][p] += 1;
		right += (l == p) ? 1 : 0;
	}
    
	PrintResult(confusion_matrix);
}

template<int InCount, int OutCount>
void StageConvolutional(
                        Layer* layer, num* buffer, 
                        double weights[InCount][OutCount][LENGTH_KERNEL][LENGTH_KERNEL],
                        double biases[OutCount])
{
	num* start = buffer + layer->networkOffset;
    
	for (int i = 0; i < OutCount; i++)
	{
		for (int j = 0; j < InCount; j++)
		{
			for (int wx = 0; wx < LENGTH_KERNEL; wx++)
			{
				for (int wy = 0; wy < LENGTH_KERNEL; wy++)
				{
					int networkIndex =
						wy +
						wx * LENGTH_KERNEL +
						j * LENGTH_KERNEL * LENGTH_KERNEL +
						i * LENGTH_KERNEL * LENGTH_KERNEL * InCount;
                    
					start[networkIndex] = (num)weights[j][i][wx][wy];
				}
			}
		}
        
		(start + layer->conv2DBiasOffset)[i] = (num)biases[i];
	}
}

template<int InCount, int OutCount>
void StageFullyConnected(
                         Layer* layer, num* buffer,
                         double weights[InCount][OutCount],
                         double biases[OutCount])
{
	num* start = buffer + layer->networkOffset;
    
	for (int i = 0; i < OutCount; i++)
	{
		for (int j = 0; j < InCount; j++)
		{
			int networkIndex = i + j * OutCount;
            
			start[networkIndex] = (num)weights[j][i];
		}
        
		(start + layer->fcBiasOffset)[i] = (num)biases[i];
	}
}

void UploadNetwork(Network* network, NetworkInstance* output, LeNet5* input)
{
	num* stage = (num*)malloc(network->networkSize * sizeof(num));
    
	StageConvolutional<INPUT, LAYER1>(&network->layers[1], stage, input->weight0_1, input->bias0_1);
	StageConvolutional<LAYER2, LAYER3>(&network->layers[3], stage, input->weight2_3, input->bias2_3);
	StageConvolutional<LAYER4, LAYER5>(&network->layers[5], stage, input->weight4_5, input->bias4_5);
	StageFullyConnected<LAYER5, OUTPUT>(&network->layers[6], stage, input->weight5_6, input->bias5_6);
    
	cudaMemcpy(output->deviceMemory, stage, network->networkSize * sizeof(num), cudaMemcpyHostToDevice);
}

template<int Count, int Width>
double CompareImages(Layer* layer, num* state, double reference[Count][Width][Width], bool print = false)
{
	num* start = state + layer->stateOffset;
    
	if (print)
	{
		// Print out
		static num buffer1[32 * 32];
		static num buffer2[32 * 32];
		num min = INFINITY;
		num max = -INFINITY;
		for (int x = 0; x < Width; x++)
		{
			for (int y = 0; y < Width; y++)
			{
				min = MIN(MIN((num)start[y + x * Width], (num)reference[0][x][y]), min);
				max = MAX(MAX((num)start[y + x * Width], (num)reference[0][x][y]), max);
			}
		}
		num rangeMultiplier = 1.0f / (max - min);
		for (int x = 0; x < Width; x++)
		{
			for (int y = 0; y < Width; y++)
			{
				buffer1[y + x * Width] = rangeMultiplier * ((num)start[y + x * Width] - min);
				buffer2[y + x * Width] = rangeMultiplier * ((num)reference[0][x][y] - min);
			}
		}
		PrintImageFloating(buffer1, layer->elementWidth, 1, false, true);
		PrintImageFloating(buffer2, layer->elementWidth, 1, false, true);
	}
    
	double deviation = 0.0f;
    
	for (int i = 0; i < Count; i++)
	{
        
		for (int x = 0; x < Width; x++)
		{
			for (int y = 0; y < Width; y++)
			{
				int stateIndex =
					y +
					x * Width +
					i * Width * Width;
                
				double current = abs((double)start[stateIndex] - reference[i][x][y]);
				if (current > deviation)
					deviation = current;
			}
		}
	}
    
	return deviation;
}

template<int Count>
double CompareLinear(Layer* layer, num* state, double reference[Count])
{
	num* start = state + layer->stateOffset;
    
	double deviation = 0.0f;
    
	for (int i = 0; i < Count; i++)
	{
		double current = abs((double)start[i] - reference[i]);
		if (current > deviation)
			deviation = current;
	}
    
	return deviation;
}

template<int InCount, int OutCount>
double CompareDeltasConvolutional(
                                  Layer* layer, num* deltas,
                                  double weights[InCount][OutCount][LENGTH_KERNEL][LENGTH_KERNEL],
                                  double biases[OutCount])
{
	num* start = deltas + layer->networkOffset;
	
	double deviation = 0.0f;
	double deviationBias = 0.0f;
    
	for (int i = 0; i < OutCount; i++)
	{
		for (int j = 0; j < InCount; j++)
		{
			for (int wx = 0; wx < LENGTH_KERNEL; wx++)
			{
				for (int wy = 0; wy < LENGTH_KERNEL; wy++)
				{
					int networkIndex =
						wy +
						wx * LENGTH_KERNEL +
						j * LENGTH_KERNEL * LENGTH_KERNEL +
						i * LENGTH_KERNEL * LENGTH_KERNEL * InCount;
                    
					double current = abs((double)start[networkIndex] - weights[j][i][wx][wy]);
                    
					if (current > deviation)
						deviation = current;
				}
			}
		}
        
		double currentBias = abs((double)(start + layer->conv2DBiasOffset)[i] - biases[i]);
		if (currentBias > deviationBias)
			deviationBias = currentBias;
	}
    
	if (deviationBias > deviation)
	{
		return -deviationBias;
	}
    
	return deviation;
}

template<int InCount, int OutCount>
double CompareDeltasFullyConnected(
                                   Layer* layer, num* deltas,
                                   double weights[InCount][OutCount],
                                   double biases[OutCount])
{
	num* start = deltas + layer->networkOffset;
    
	double deviation = 0.0f;
	double deviationBias = 0.0f;
    
	for (int i = 0; i < OutCount; i++)
	{
		for (int j = 0; j < InCount; j++)
		{
			double current = abs((double)start[i + j * OutCount] - weights[j][i]);
			if (current > deviation)
				deviation = current;
		}
        
		double currentBias = abs((double)(start + layer->fcBiasOffset)[i] - biases[i]);
		if (currentBias > deviationBias)
			deviationBias = currentBias;
	}
    
	if (deviationBias > deviation)
	{
		return -deviationBias;
	}
    
	return deviation;
}

#define LAYER_TEST(text, call) { double d = (call); printf("%s: ", text); if (abs(d)>0.0001) { printf("FAIL"); } else { printf("PASS"); } if (d < 0) { printf(" (%f B)\n", -d); } else { printf(" (%f)\n", d); } }

void RunTest(Dataset datasetTrain, Dataset datasetTest)
{
	// Initialize reference
    
	LeNet5* lenet = (LeNet5*)malloc(sizeof(LeNet5));
	Initial(lenet);
	Feature refFeatures = { 0 };
	Feature refErrors = { 0 };
	LeNet5 refDeltas = { 0 };
    
    
    
	// Initialize the CNN
	
	int batchSize = 1;
    
	Network network = {};
	NetworkCreateLeNet5Reference(&network);
	
	NetworkInstance instance;
	NetworkInstanceCreate(&network, &instance, false);
    
	NetworkState state;
	NetworkState errors;
	NetworkStateCreate(&network, &state, 1);
	NetworkStateCreate(&network, &errors, 1);
    
	num* stateBuffer = (num*)malloc(network.stateSize * batchSize * sizeof(num));
	num* deltaBuffer = (num*)malloc(network.networkSize * batchSize * sizeof(num));
    
	num* deltas;
	cudaMalloc(&deltas, network.networkSize * batchSize * sizeof(num));
    
    
    
	// Copy the reference LeNet5 to device memory
    
	UploadNetwork(&network, &instance, lenet);
    
    
    
	// Run the networks
    
	DebugForward(lenet, &refFeatures, ((image*)datasetTrain.images)[0]);
    
	NetworkForwardPropagation(&network, &instance, &state, &datasetTrain, 0, 1);
    
    
    
	// Compare the results
    
	cudaMemcpy(stateBuffer, state.deviceMemory, network.stateSize * sizeof(num), cudaMemcpyDeviceToHost);
    
	LAYER_TEST("State Input Layer  ", (CompareImages<1, 32>(&network.layers[0], stateBuffer, refFeatures.input)));
	LAYER_TEST("State Conv Layer 1 ", (CompareImages<6, 28>(&network.layers[1], stateBuffer, refFeatures.layer1)));
	LAYER_TEST("State Pool Layer 1 ", (CompareImages<6, 14>(&network.layers[2], stateBuffer, refFeatures.layer2)));
	LAYER_TEST("State Conv Layer 2 ", (CompareImages<16, 10>(&network.layers[3], stateBuffer, refFeatures.layer3)));
	LAYER_TEST("State Pool Layer 2 ", (CompareImages<16, 5>(&network.layers[4], stateBuffer, refFeatures.layer4)));
	LAYER_TEST("State Conv Layer 3 ", (CompareImages<120, 1>(&network.layers[5], stateBuffer, refFeatures.layer5)));
	LAYER_TEST("State FC Layer     ", (CompareLinear<10>(&network.layers[6], stateBuffer, refFeatures.output)));
	printf("\n");
    
    
    
	// Run the networks
    
	DebugBackward(lenet, &refFeatures, &refErrors, &refDeltas, datasetTrain.labels[0]);
    
	NetworkLoadOutputSoftmax(&network, &state, &errors, datasetTrain.labels, batchSize);
	NetworkBackwardPropagation(&network, &instance, &state, &errors, deltas, batchSize);
    
    
    
	// Compare the results
    
	cudaMemcpy(stateBuffer, errors.deviceMemory, network.stateSize * sizeof(num), cudaMemcpyDeviceToHost);
	cudaMemcpy(deltaBuffer, deltas, network.networkSize * sizeof(num), cudaMemcpyDeviceToHost);
    
	LAYER_TEST("Error FC Layer     ", (CompareLinear<10>(&network.layers[6], stateBuffer, refErrors.output)));
	LAYER_TEST("Error Conv Layer 3 ", (CompareImages<120, 1>(&network.layers[5], stateBuffer, refErrors.layer5)));
	LAYER_TEST("Error Pool Layer 2 ", (CompareImages<16, 5>(&network.layers[4], stateBuffer, refErrors.layer4, true)));
	LAYER_TEST("Error Conv Layer 2 ", (CompareImages<16, 10>(&network.layers[3], stateBuffer, refErrors.layer3, true)));
	LAYER_TEST("Error Pool Layer 1 ", (CompareImages<6, 14>(&network.layers[2], stateBuffer, refErrors.layer2, true)));
	LAYER_TEST("Error Conv Layer 1 ", (CompareImages<6, 28>(&network.layers[1], stateBuffer, refErrors.layer1)));
	printf("\n");
    
	LAYER_TEST("Deltas FC Layer     ", (CompareDeltasFullyConnected<LAYER5, OUTPUT>(&network.layers[6], deltaBuffer, refDeltas.weight5_6, refDeltas.bias5_6)));
	LAYER_TEST("Deltas Conv Layer 3 ", (CompareDeltasConvolutional<LAYER4, LAYER5>(&network.layers[5], deltaBuffer, refDeltas.weight4_5, refDeltas.bias4_5)));
	LAYER_TEST("Deltas Conv Layer 2 ", (CompareDeltasConvolutional<LAYER2, LAYER3>(&network.layers[3], deltaBuffer, refDeltas.weight2_3, refDeltas.bias2_3)));
	LAYER_TEST("Deltas Conv Layer 1 ", (CompareDeltasConvolutional<INPUT, LAYER1>(&network.layers[1], deltaBuffer, refDeltas.weight0_1, refDeltas.bias0_1)));
	printf("\n");
}

//~ Run and test

static void RunCNN(Dataset datasetTrain, Dataset datasetTest)
{
	// Tweakable parameters
	const int batchSize = 300;
	const num alpha = (num)0.5;
    
    printf("Clocks per second: %u\n\n", (unsigned int)CLOCKS_PER_SEC);
    
    // Create the network structure
	Network network = {};
	NetworkCreateLeNet5Reference(&network);
    
	// Create an instance of the network
	NetworkInstance instance;
	NetworkInstanceCreate(&network, &instance, true);
    
	// Train the network
    clock_t time = clock();
	NetworkTrain(&network, &instance, &datasetTrain, batchSize, alpha);
    time = clock() - time;
    printf("Train time: %.3f seconds\n\n", ((double)time) / CLOCKS_PER_SEC);
    
	// Test the network
    time = clock();
	NetworkTest(&network, &instance, &datasetTest, true);
    time = clock() - time;
    printf("\nTest time: %.3f seconds\n", ((double)time) / CLOCKS_PER_SEC);
    
    // Run an extended test
    int extendedCount = 20;
    printf("\nRunning extended test");
    time = clock();
	for (int runs = 0; runs < extendedCount; runs++)
	{
        NetworkTrain(&network, &instance, &datasetTrain, batchSize + runs * 10, alpha * (float)(extendedCount - runs) / extendedCount);
        printf(".");
        fflush(stdout);
	}
    time = clock() - time;
    double elapsed = ((double)time) / CLOCKS_PER_SEC;
    printf("\nExtended time: %.3f seconds. Average: %.3f seconds.\n\n", elapsed, elapsed / extendedCount);
    
    // Test the network
    NetworkTest(&network, &instance, &datasetTest, true);
    
	// Deinitialize
	NetworkInstanceDestroy(&instance);
	NetworkDestroy(&network);
}

static void RunReference(Dataset datasetTrain, Dataset datasetTest)
{
    
}



//~ Entry point

int main()
{
	// Import data
	Dataset datasetTrain;
	Dataset datasetTest;
	ReadDataset(&datasetTrain, COUNT_TRAIN, IMAGE_WIDTH, IMAGE_WIDTH, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL);
	ReadDataset(&datasetTest, COUNT_TEST, IMAGE_WIDTH, IMAGE_WIDTH, FILE_TEST_IMAGE, FILE_TEST_LABEL);
    
	//RunTest(datasetTrain, datasetTest);
    
    RunCNN(datasetTrain, datasetTest);
	
	return 0;
}
