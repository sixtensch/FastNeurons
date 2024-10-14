#pragma once

#define DEBUG_PRINT 0



//~ Basic data type definitions

#define num float
#define byte unsigned char

typedef byte Pixel;
typedef byte Label;



//~ Function pointer type definitions

typedef num(*Activation)(num);



//~ Suport struct definitions

struct Dataset
{
	int count;
	Pixel* images;
	Label* labels;
};

enum LayerType
{
	LayerTypeInput2D,
	LayerTypeConvolution2D,
	LayerTypePooling2D,
	LayerTypeFullyConnected
};

enum PoolingType
{
	PoolingTypeMax,
	PoolingTypeAverage
};

enum ActivationType
{
	ActivationTypeReLU,
	ActivationTypePReLU,
	ActivationTypeTanh,
	ActivationTypeGaussian,
};

struct Layer
{
	LayerType type;

	int networkOffset; // Where weights and balances are stored
	int networkSize;

	int stateOffset;   // Where working memory is stored
	int stateSize;

	// What is produced by the layer
	int elementCount;
	int elementWidth;
	int elementHeight;

	union
	{
		struct
		{
			int input2DPadding; // How many pixels smaller is the actual input
		};

		struct
		{
			ActivationType conv2DActivationType;
			int conv2DKernelWidth; // Always square
			int conv2DBiasOffset;
		};

		struct
		{
			int pool2DEyeWidth; // Always square
			PoolingType pool2DType;
		};

		struct
		{
			ActivationType fcActivationType;
			int fcBiasOffset;
		};
	};
};



//~ Main types

// Describes a network and how it is structured
struct Network
{
	Layer* layers;
	int layerCount;
	int layerCapacity;

	int networkSize;  // Weights, etc
	int stateSize;    // Active features
};

// Stores information about an instance of a network, such as weights and biases
struct NetworkInstance
{
	num* deviceMemory;
};

// Stores working information used by a network, such as input, output, and intermediate information
struct NetworkState
{
	int count; // Number of network states
	num* deviceMemory;
};



//~ Data I/O functions

// Read image and label files into computer memory
int ReadDataset(Dataset* dataset, int count, int width, int height, const char* pathImages, const char* pathLabels);

// Print one image to standard output
void PrintImage(Pixel* pixels, int width, int sampling, bool tendToMax, bool drawFrame);
void PrintImageFloating(num* pixels, int width, int sampling, bool tendToMax, bool drawFrame);

void PrintResult(int confusion_matrix[10][10]);



//~ Network management functions

// Create and destroy a neural network
void NetworkCreate(Network* network, int layerCapacity);
void NetworkDestroy(Network* network);

// Addd layers to a neural network after creation
void NetworkAddLayerInput2D(Network* network, int count, int width, int height, int padding);
void NetworkAddLayerConvolution2D(Network* network, int count, int kernelWidth, ActivationType activation);
void NetworkAddLayerPooling2D(Network* network, int eyeWidth, PoolingType pooling);
void NetworkAddLayerFullyConnected(Network* network, int count, ActivationType activation);

// Create and destroy memory instances of a neural network
void NetworkInstanceCreate(Network* network, NetworkInstance* instance, bool random);
void NetworkInstanceDestroy(NetworkInstance* instance);

// Create and destroy active memory states of a neural network
void NetworkStateCreate(Network* network, NetworkState* state, int count);
void NetworkStateDestroy(NetworkState* state);



//~ Network usage functions

// Train a network instance once on a given dataset using forward and backward propagation
void NetworkTrain(Network* network, NetworkInstance* instance, Dataset* dataset, int desiredBatchSize, num alpha);

// Test a neural network against a control dataset
void NetworkTest(Network* network, NetworkInstance* instance, Dataset* dataset, bool printMatrix);



//~ Advanced functions

void NetworkForwardPropagation(Network* network, NetworkInstance* instance, NetworkState* state, Dataset* dataset, int datasetOffset, int batchSize);
void NetworkLoadOutputSoftmax(Network* network, NetworkState* state, NetworkState* errors, Label* correct, int batchSize, Label* labelBuffer = nullptr);
void NetworkBackwardPropagation(Network* network, NetworkInstance* instance, NetworkState* state, NetworkState* errors, num* deltas, int batchSize);
void NetworkApplyDeltas(Network* network, NetworkInstance* instance, num* deltas, num alpha, int batchSize);
