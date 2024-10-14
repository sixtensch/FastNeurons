#include <cnn.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

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

void NetworkCreateLeNet5(Network* network)
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



//~ Run and test

static void RunCNN(Dataset datasetTrain, Dataset datasetTest)
{
	// Tweakable parameters
	const int batchSize = 300;
	const num alpha = (num)0.5;
    
    printf("Clocks per second: %u\n\n", (unsigned int)CLOCKS_PER_SEC);
    
    // Create the network structure
	Network network = {};
	NetworkCreateLeNet5(&network);
    
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



//~ Entry point

int main()
{
	// Import data
	Dataset datasetTrain;
	Dataset datasetTest;
	ReadDataset(&datasetTrain, COUNT_TRAIN, IMAGE_WIDTH, IMAGE_WIDTH, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL);
	ReadDataset(&datasetTest, COUNT_TEST, IMAGE_WIDTH, IMAGE_WIDTH, FILE_TEST_IMAGE, FILE_TEST_LABEL);
    
    RunCNN(datasetTrain, datasetTest);
	
	return 0;
}
