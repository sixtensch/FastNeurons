
//~ Includes

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <ctime>

#include <random>
#include <chrono>



//~ Training data definition

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE  "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL  "t10k-labels-idx1-ubyte"

#define COUNT_TRAIN 60000
#define COUNT_TEST  10000

#define IMAGE_WIDTH 28
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_WIDTH)



//~ Macro definitions

// Column major get
#define GET(data, x, y, height) (data[x * height + y])

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define PRINT_ERROR(format, ...)   printf("Error:   " format "\n" __VA_OPT__(,) __VA_ARGS__);
#define PRINT_WARNING(format, ...) printf("Warning: " format "\n" __VA_OPT__(,) __VA_ARGS__);
#define PRINT_INFO(format, ...)    printf("Info:    " format "\n" __VA_OPT__(,) __VA_ARGS__);

#define CUDA_CHECK(stamp) { cudaError_t _e = cudaGetLastError(); if (_e != cudaSuccess) { PRINT_ERROR("%i at [" stamp "]: %s", _e, cudaGetErrorString(_e)); fflush(stdout); } }



//~ Data typedefs

typedef float          num;
typedef unsigned char  byte;

typedef byte Pixel;
typedef byte Label;



//~ Struct definitions

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



//~ Math functions

__device__ num ReLU(num x)
{
    return x * (x > 0);
}

__device__ num ReLUDeriv(num x)
{
    return x > 0;
}

#define PRELU_A ((num)0.02)
__device__ num PReLU(num x)
{
    bool p = x > 0;
    return 
        x * p + 
        x * PRELU_A * !p;
}

__device__ num PReLUDeriv(num x)
{
    bool p = x > 0;
    return p + !p * PRELU_A;
}

num RandomNum(num min, num max)
{
    static std::mt19937 engine(time(0));
    
    std::uniform_real_distribution<num> distribution(min, max);
    return distribution(engine);
}



//~ Help functions

int ReadDataset(Dataset* dataset, int count, int width, int height, const char* pathImages, const char* pathLabels)
{
    FILE* fileImage = fopen(pathImages, "rb");
    FILE* fileLabel = fopen(pathLabels, "rb");
    
    if (fileImage == nullptr || fileLabel == nullptr)
    {
        return 1;
    }
    
    dataset->count = count;
    dataset->images = (Pixel*)malloc(count * width * height * sizeof(Pixel));
    dataset->labels = (Label*)malloc(count * sizeof(Label));
    
	fseek(fileImage, 16, SEEK_SET);
	fseek(fileLabel, 8, SEEK_SET);
    
	fread(dataset->images, count * width * height * sizeof(Pixel), 1, fileImage);
	fread(dataset->labels, count * sizeof(Label), 1, fileLabel);
    
	fclose(fileImage);
	fclose(fileLabel);
    
    return 0;
}

// Provided with the example
void PrintResult(int confusion_matrix[10][10])
{
	// Print the confusion matrix
	printf("%15sPredicted label\n%10s", " ", " ");
	for (int col = 0; col < 10; col++)
		printf("%6d", col);
	printf("%10s\n", "Total");
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\nTrue label\n");
	int row_labels = 0;
	int total = 0;
	for (int row = 0; row < 10; row++) {
		row_labels = 0;
		printf("%10d", row);
		for (int col = 0; col < 10; col++) {
			printf("%6d", confusion_matrix[row][col]);
			row_labels += confusion_matrix[row][col];
		}
		printf("%10d\n", row_labels);
		total += row_labels;
	}
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\n%67s = %10d\n", "Total number of input images tested", total);
	for (int n = 0; n < 70; n++)
		printf("%s", "-");
	printf("\n");
}


void PrintImage(Pixel* pixels, int width, int sampling = 2, bool tendToMax = true, bool drawFrame = true)
{
    static const char characters[] = " .*#";
    static const byte characterCount = sizeof(characters) - 1;
    
    static char buffer[(32 * 2 + 3) * (32 + 2) + 1] = { '\0' };
    int i = 0;
    
#define BPRINT(c) buffer[i] = c; i++;
    
    if (drawFrame)
    {
        BPRINT('*');
        for (int f = 0; f < width / sampling; f++) 
        {
            BPRINT('-');
            BPRINT('-');
        }
        BPRINT('*');
        BPRINT('\n');
    }
    
    for (int xp = 0; xp < width / sampling; xp++)
    {
        if (drawFrame)
        {
            BPRINT('|');
        }
        
        for (int yp = 0; yp < width / sampling; yp++)
        {
            unsigned int max = 0;
            unsigned int sum = 0;
            
            for (int xs = 0; xs < sampling; xs++)
            {
                for (int ys = 0; ys < sampling; ys++)
                {
                    int x = xp * sampling + xs;
                    int y = yp * sampling + xs;
                    
                    sum += GET(pixels, x, y, width);
                    
                    Pixel current = GET(pixels, x, y, width);
                    if (current > max)
                    {
                        max = current;
                    }
                }
            }
            
            Pixel value = 0;
            unsigned int average = sum / (sampling * sampling);
            
            if (tendToMax)
            {
                // Middle point between average and max, gives good legibility
                value = (Pixel)((average + max) / 2);
            }
            else
            {
                value = (Pixel)average;
            }
            
            BPRINT(characters[value / (256 / characterCount)]);
            BPRINT(characters[value / (256 / characterCount)]);
        }
        
        if (drawFrame)
        {
            BPRINT('|');
        }
        
        BPRINT('\n');
    }
    
    if (drawFrame)
    {
        BPRINT('*');
        for (int f = 0; f < width / sampling; f++) 
        {
            BPRINT('-');
            BPRINT('-');
        }
        BPRINT('*');
        BPRINT('\n');
    }
    
    buffer[i] = '\0';
    
#undef BPRINT
    
    printf("%s", buffer);
}

void PrintImageFloating(num* pixels, int width, int sampling = 2, bool tendToMax = true, bool drawFrame = true)
{
    Pixel* buffer = (Pixel*)malloc(width * width * sizeof(Pixel));
    
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < width; y++)
        {
            GET(buffer, x, y, width) = (Pixel)(MAX(MIN((int)(255 * GET(pixels, x, y, width)), 255), 0));
        }
    }
    
    PrintImage(buffer, width, sampling, tendToMax, drawFrame);
    
    free(buffer);
}



//~ Network help functions

int NetworkAddLayerHelper(Network* network, int networkSize, int stateSize)
{
    if (network->layerCount >= network->layerCapacity)
    {
        PRINT_ERROR("Maximum number of layers reached.\n");
        return -1;
    }
    
    int index = network->layerCount;
    
    network->networkSize += networkSize; 
    network->stateSize += stateSize; 
    
    network->layers[index].networkSize = networkSize;
    network->layers[index].stateSize = stateSize;
    
    network->layers[index].elementCount = 1;
    network->layers[index].elementWidth = 1;
    network->layers[index].elementHeight = 1;
    
    if (network->layerCount > 0)
    {
        network->layers[index].networkOffset = 
            network->layers[index - 1].networkOffset + 
            network->layers[index - 1].networkSize;
        
        network->layers[index].stateOffset = 
            network->layers[index - 1].stateOffset + 
            network->layers[index - 1].stateSize;
    }
    else
    {
        network->layers[index].networkOffset = 0;
        network->layers[index].stateOffset   = 0;
    }
    
    network->layerCount++;
    
    return index;
}



//~ Network creation functions

void NetworkCreate(Network* network, int layerCapacity)
{
    network->layers = (Layer*)malloc(layerCapacity * sizeof(Layer));
    memset(network->layers, 0, layerCapacity * sizeof(Layer));
    
    network->networkSize = 0;
    network->stateSize = 0;
    
    network->layerCount = 0;
    network->layerCapacity = layerCapacity;
}

void NetworkDestroy(Network* network)
{
    free(network->layers);
}

void NetworkAddLayerInput2D(Network* network, int count, int width, int height, int padding)
{
    int networkSize = 0;
    int stateSize = count * (width + padding) * (height + padding);
    
    int index = NetworkAddLayerHelper(network, networkSize, stateSize);
    if (index < 0)
    {
        return;
    }
    
    network->layers[index].type = LayerTypeInput2D;
    
    network->layers[index].elementCount = count;
    network->layers[index].elementWidth = width + padding;
    network->layers[index].elementHeight = height + padding;
    
    network->layers[index].input2DPadding = padding;
}

void NetworkAddLayerConvolution2D(Network* network, int count, int kernelWidth, ActivationType activation)
{
    if (network->layerCount < 1 || 
        network->layers[network->layerCount - 1].elementWidth < kernelWidth || 
        network->layers[network->layerCount - 1].elementHeight < kernelWidth)
    {
        PRINT_ERROR("Convolution layer incompatible with previous layer.\n");
        return;
    }
    
    int oldCount = network->layers[network->layerCount - 1].elementCount;
    int width = network->layers[network->layerCount - 1].elementWidth - kernelWidth + 1;
    int height = network->layers[network->layerCount - 1].elementHeight - kernelWidth + 1;
    
    int weightsSize = oldCount * count * kernelWidth * kernelWidth;
    
    int networkSize = weightsSize + count; // Weigths + biases
    int stateSize = count * width * height;
    
    int index = NetworkAddLayerHelper(network, networkSize, stateSize);
    if (index < 0)
    {
        return;
    }
    
    network->layers[index].type = LayerTypeConvolution2D;
    
    network->layers[index].elementCount = count;
    network->layers[index].elementWidth = width;
    network->layers[index].elementHeight = height;
    
    network->layers[index].conv2DActivationType = activation;
    network->layers[index].conv2DKernelWidth = kernelWidth;
    network->layers[index].conv2DBiasOffset = weightsSize;
}

void NetworkAddLayerPooling2D(Network* network, int eyeWidth, PoolingType pooling)
{
    if (network->layerCount < 1 || 
        network->layers[network->layerCount - 1].elementWidth < eyeWidth || 
        network->layers[network->layerCount - 1].elementHeight < eyeWidth)
    {
        PRINT_ERROR("Pooling layer incompatible with previous layer.\n");
        return;
    }
    
    if (network->layers[network->layerCount - 1].elementWidth % eyeWidth > 0 || 
        network->layers[network->layerCount - 1].elementHeight % eyeWidth > 0)
    {
        PRINT_WARNING("Pooling layer will delete information.\n");
    }
    
    int count = network->layers[network->layerCount - 1].elementCount;
    int width = network->layers[network->layerCount - 1].elementWidth / eyeWidth;
    int height = network->layers[network->layerCount - 1].elementHeight / eyeWidth;
    
    int networkSize = 0;
    int stateSize = count * width * height;
    
    int index = NetworkAddLayerHelper(network, networkSize, stateSize);
    if (index < 0)
    {
        return;
    }
    
    network->layers[index].type = LayerTypePooling2D;
    
    network->layers[index].elementCount = count;
    network->layers[index].elementWidth = width;
    network->layers[index].elementHeight = height;
    
    network->layers[index].pool2DEyeWidth = eyeWidth;
    network->layers[index].pool2DType = pooling;
}

void NetworkAddLayerFullyConnected(Network* network, int count, ActivationType activation)
{
    if (network->layerCount < 1)
    {
        PRINT_ERROR("Fully connected layer cannot be first layer.\n");
        return;
    }
    
    int oldCountTotal = 
        network->layers[network->layerCount - 1].elementCount *
        network->layers[network->layerCount - 1].elementWidth *
        network->layers[network->layerCount - 1].elementHeight;
    
    int weightsSize = oldCountTotal * count;
    
    int networkSize = weightsSize + count; // Weights + biases
    int stateSize = count;
    
    int index = NetworkAddLayerHelper(network, networkSize, stateSize);
    if (index < 0)
    {
        return;
    }
    
    network->layers[index].type = LayerTypeFullyConnected;
    
    network->layers[index].elementCount = count;
    network->layers[index].elementWidth = 1;
    network->layers[index].elementHeight = 1;
    
    network->layers[index].fcActivationType = activation;
    network->layers[index].fcBiasOffset = weightsSize;
}



//~ Initializaiton functions

void NetworkCreateLeNet5Reference(Network* network)
{
    const int kernelWidth = 5;
    
    NetworkCreate(network, 7);
    
    // Add the layers
    NetworkAddLayerInput2D        (network, 1, 28, 28, 4);
    NetworkAddLayerConvolution2D  (network, 6, kernelWidth, ActivationTypeReLU);
    NetworkAddLayerPooling2D      (network, 2, PoolingTypeMax);
    NetworkAddLayerConvolution2D  (network, 16, kernelWidth, ActivationTypeReLU);
    NetworkAddLayerPooling2D      (network, 2, PoolingTypeMax);
    NetworkAddLayerConvolution2D  (network, 120, kernelWidth, ActivationTypeReLU);
    NetworkAddLayerFullyConnected (network, 10, ActivationTypeReLU);
}

void NetworkCreateLeNet5Real(Network* network)
{
    const int kernelWidth = 5;
    
    NetworkCreate(network, 8);
    
    // Add the layers
    NetworkAddLayerInput2D        (network, 1, 28, 28, 4);
    NetworkAddLayerConvolution2D  (network, 6, kernelWidth, ActivationTypeReLU);
    NetworkAddLayerPooling2D      (network, 2, PoolingTypeMax);
    NetworkAddLayerConvolution2D  (network, 16, kernelWidth, ActivationTypeReLU);
    NetworkAddLayerPooling2D      (network, 2, PoolingTypeMax);
    NetworkAddLayerFullyConnected (network, 120, ActivationTypeReLU);
    NetworkAddLayerFullyConnected (network, 84, ActivationTypeReLU);
    NetworkAddLayerFullyConnected (network, 10, ActivationTypeReLU);
}

void NetworkCreateSimple(Network* network)
{
    NetworkCreate(network, 4);
    
    // Add the layers
    NetworkAddLayerInput2D        (network, 1, 28, 28, 0);
    NetworkAddLayerFullyConnected (network, 16, ActivationTypeReLU);
    NetworkAddLayerFullyConnected (network, 16, ActivationTypeReLU);
    NetworkAddLayerFullyConnected (network, 10, ActivationTypeReLU);
}

void NetworkCreateNNADL(Network* network)
{
    NetworkCreate(network, 4);
    
    // Add the layers
    NetworkAddLayerInput2D        (network, 1, 28, 28, 0);
    NetworkAddLayerFullyConnected (network, 30, ActivationTypeReLU);
    NetworkAddLayerFullyConnected (network, 10, ActivationTypeReLU);
}

void NetworkInstanceCreate(Network* network, NetworkInstance* instance, bool random = true)
{
    cudaMalloc(&instance->deviceMemory, network->networkSize * sizeof(num));
    
    CUDA_CHECK("NetworkInstanceCreate Allocation");
    
    if (!random)
    {
        return;
    }
    
    num* stage = (num*)malloc(network->networkSize * sizeof(num));
    memset(stage, 0, network->networkSize * sizeof(num));
    
    Layer* layer = nullptr;
    Layer* previous = nullptr;
    
    for (int i = 0; i < network->layerCount; i++)
    {
        layer = &network->layers[i];
        
        switch (layer->type)
        {
            case LayerTypeConvolution2D:
            {
                num factor = (num)sqrt(6.0 / (layer->conv2DKernelWidth *
                                              layer->conv2DKernelWidth * 
                                              (previous->elementCount + layer->elementCount)));
                
                for (int j = 0; j < layer->conv2DBiasOffset; j++)
                {
                    stage[layer->networkOffset + j] = factor * RandomNum(-1, 1);
                }
            }
            break;
            
            case LayerTypeFullyConnected:
            {
                num factor = (num)sqrt(6.0 / (previous->elementCount *
                                              previous->elementWidth * 
                                              previous->elementHeight + 
                                              layer->elementCount));
                
                for (int j = 0; j < layer->fcBiasOffset; j++)
                {
                    stage[layer->networkOffset + j] = factor * RandomNum(-1, 1);
                }
            }
            break;
        }
        
        previous = layer;
    }
    
    cudaMemcpy(instance->deviceMemory, stage, network->networkSize * sizeof(num), cudaMemcpyHostToDevice);
    
    CUDA_CHECK("NetworkInstanceCreate Memcpy");
    
    free(stage);
}

void NetworkInstanceDestroy(NetworkInstance* instance)
{
    cudaFree(instance->deviceMemory);
    instance->deviceMemory = nullptr;
    
    CUDA_CHECK("NetworkInstanceDestroy");
}

void NetworkStateCreate(Network* network, NetworkState* state, int count)
{
    state->count = count;
    cudaMalloc(&state->deviceMemory, count * network->stateSize * sizeof(num));
    
    CUDA_CHECK("NetworkStateCreate");
}

void NetworkStateDestroy(NetworkState* state)
{
    cudaFree(state->deviceMemory);
    state->count = 0;
    state->deviceMemory = nullptr;
    
    CUDA_CHECK("NetworkStateDestroy");
}



//- Kernels

//~ Forward propagation kernels

template<auto Activation>
__global__ void Convolve2DForward(num* weights, num* biases, num* input, num* output, int stateSize, 
                                  int inWidth, int inHeight, int inCount, 
                                  int outWidth, int outHeight, int outCount,
                                  int kernelWidth)
{
    num accumulator = 0;
    
    int kernelSize = kernelWidth * kernelWidth;
    int inputSize = inWidth * inHeight;
    int outputSize = outWidth * outHeight;
    
    //int kernelOffset = threadIdx.x * kernelWidth + threadIdx.y + kernelSize * outCount * blockIdx.x;
    
    for (int i = 0; i < inCount; i++)
    {
        for (int x = 0; x < kernelWidth; x++)
        {
            for (int y = 0; y < kernelWidth; y++)
            {
                // Kernels are serialized by index, where all the weights at [0, 0] are adjacent
                num weight = weights[threadIdx.x * kernelWidth + 
                                     threadIdx.y + 
                                     blockIdx.x * kernelSize * outCount + 
                                     kernelSize * (x * kernelWidth + y)];
                
                // Images are serialized column-major in series
                accumulator += weight * input[(threadIdx.x + x) * inHeight + 
                                              (threadIdx.y + y) + 
                                              i * inputSize +
                                              blockIdx.y * stateSize];
            }
        }
    }
    
    num result = Activation(accumulator + biases[blockIdx.x]);
    
    output[threadIdx.x * outHeight + 
           threadIdx.y + 
           outputSize * blockIdx.x +
           stateSize * blockIdx.y] = result;
}

__global__ void Pool2DForwardMax(num* input, num* output, int stateSize, int count,
                                 int inWidth, int inHeight, int outWidth, int outHeight, int eyeWidth)
{
    num max = 0;
    
    int inputSize = inWidth * inHeight;
    int outputSize = outWidth * outHeight;
    
    int inputOffset = inputSize * blockIdx.x + stateSize * blockIdx.y;
    
    for (int x = 0; x < eyeWidth; x++)
    {
        for (int y = 0; y < eyeWidth; y++)
        {
            num value = input[(threadIdx.x * eyeWidth + x) * inHeight +
                              (threadIdx.y * eyeWidth + y) +
                              inputOffset];
            
            bool greater = value > max;
            max = greater * value + !greater * max; 
        }
    }
    
    output[threadIdx.x * outHeight +
           threadIdx.y + 
           outputSize * blockIdx.x +
           stateSize * blockIdx.y] = max;
}

template<auto Activation>
__global__ void FullyConnectForward(num* weights, num* biases, num* input, num* output, int stateSize,
                                    int inCount, int outCount)
{
    num accumulator = 0;
    
    for (int i = 0; i < inCount; i++)
    {
        accumulator += input[i + stateSize * blockIdx.x] * weights[threadIdx.x + i * blockDim.x];
    }
    
    num result = Activation(accumulator + biases[threadIdx.x]);
    
    output[threadIdx.x + stateSize * blockIdx.x] = result;
}

//~ Backpropagation kernels

// Initialization for output layers
// Computes error based on a type of desired value (cost)

__global__ void SoftmaxComputeErrors(num* output, num* errors, Label* labels, int stateSize, int count)
{
    __shared__ num inner;
    
    int label = (int)labels[blockIdx.x];
    
    num sum = 0;
    num myValue = output[threadIdx.x + blockIdx.x * stateSize];
    for (int i = 0; i < count; i++)
    {
        sum += expf(output[i + blockIdx.x * stateSize] - myValue);
    }
    
    num error = (num)1.0 / sum;
    
    if (label == threadIdx.x)
    {
        inner = error;
    }
    
    __syncthreads();
    
    atomicAdd(&inner, -(error * error));
    
    __syncthreads();
    
    errors[threadIdx.x + blockIdx.x * stateSize] += error * ((threadIdx.x == label) - error - inner);
}

// Layer backpropagation
// Calculates error for input layer based on error of output layer, as well as connecting deltas for weights and biases

template<auto ActivationDeriv>
__global__ void BackwardActivation(num* input, num* inErrors, int stateSize, int blockSize)
{
    int id = threadIdx.x * blockDim.y + threadIdx.y;
    
    int index = id +
        blockIdx.x * blockSize +
        blockIdx.y * stateSize;
    
    inErrors[index] *= ActivationDeriv(input[index]); 
}

__global__ void Convolve2DBackwardSum(num* weights, num* inErrors, num* outErrors, int stateSize, 
                                      int inWidth, int inHeight, int inCount, 
                                      int outWidth, int outHeight, int outCount,
                                      int kernelWidth)
{
    int kernelSize = kernelWidth * kernelWidth;
    int inputSize = inWidth * inHeight;
    int outputSize = outWidth * outHeight;
    
    //int kernelOffset = threadIdx.x * kernelWidth + threadIdx.y;
    
    for (int i = 0; i < outCount; i++)
    {
        for (int x = 0; x < kernelWidth; x++)
        {
            for (int y = 0; y < kernelWidth; y++)
            {
                num weight = weights[threadIdx.x * kernelWidth +
                                     threadIdx.y +
                                     i * kernelSize * outCount +
                                     kernelSize * (x * kernelWidth + y)];
                
                num value = weight * outErrors[threadIdx.x * outHeight + 
                                               threadIdx.y +
                                               i * outputSize +
                                               blockIdx.y * stateSize];
                
                inErrors[(threadIdx.x + x) * inHeight + 
                         (threadIdx.y + y) +
                         blockIdx.x * inputSize +
                         blockIdx.y * stateSize] += value;
            }
        }
    }
}

__global__ void Convolve2DBackwardBiases(num* outErrors, num* biasDeltas, int stateSize, int networkSize, int outWidth, int outHeight)
{
    num columnSum = 0;
    
    for (int y = 0; y < outHeight; y++)
    {
        columnSum += outErrors[threadIdx.x * outHeight + y + 
                               blockIdx.x * outWidth * outHeight + 
                               blockIdx.y * stateSize]; 
    }
    
    atomicAdd(&biasDeltas[blockIdx.x + blockIdx.y * networkSize], columnSum);
}

__global__ void Convolve2DBackwardWeights(num* input, num* outErrors, num* weightDeltas, int stateSize, int networkSize,
                                          int inWidth, int inHeight, int inCount, 
                                          int outWidth, int outHeight, int outCount,
                                          int kernelWidth)
{
    num accumulator = 0;
    
    int kernelSize = kernelWidth * kernelWidth;
    int inputSize = inWidth * inHeight;
    int outputSize = outWidth * outHeight;
    
    //int kernelOffset = threadIdx.x * kernelWidth + threadIdx.y + kernelSize * outCount * blockIdx.x;
    
    for (int i = 0; i < inCount; i++)
    {
        for (int x = 0; x < outWidth; x++)
        {
            for (int y = 0; y < outHeight; y++)
            {
                // Kernels are serialized by index, where all the outErrors at [0, 0] are adjacent
                num weight = outErrors[x * outHeight + y + 
                                       blockIdx.x * outputSize + 
                                       blockIdx.y * stateSize];
                
                // Images are serialized column-major in series
                accumulator += weight * input[(threadIdx.x + x) * inHeight + 
                                              (threadIdx.y + y) + 
                                              i * inputSize +
                                              blockIdx.y * stateSize];
            }
        }
    }
    
    weightDeltas[threadIdx.x * kernelWidth + 
                 threadIdx.y + 
                 blockIdx.x * kernelSize +
                 blockIdx.y * networkSize] += accumulator;
}

__global__ void Pool2DBackwardMax(num* input, num* inErrors, num* outErrors, int stateSize, int count,
                                  int inWidth, int inHeight, int outWidth, int outHeight, int eyeWidth)
{
    int xMax = 0;
    int yMax = 0;
    
    int inSize = inWidth * inHeight;
    
    for (int x = 0; x < eyeWidth; x++)
    {
        for (int y = 0; y < eyeWidth; y++)
        {
            num current = input[(threadIdx.x * eyeWidth + x) * inHeight +
                                (threadIdx.y * eyeWidth + y) +
                                blockIdx.x * inSize +
                                blockIdx.y * stateSize];
            
            num comp = input[(threadIdx.x * eyeWidth + xMax) * inHeight +
                             (threadIdx.y * eyeWidth + yMax) +
                             blockIdx.x * inSize +
                             blockIdx.y * stateSize];
            
            bool greater = current > comp;
            
            xMax += greater * (x - xMax);
            yMax += greater * (y - yMax);
        }
    }
    
    inErrors[(threadIdx.x * eyeWidth + xMax) * inHeight +
             (threadIdx.y * eyeWidth + yMax) +
             blockIdx.x * inSize +
             blockIdx.y * stateSize] += outErrors[threadIdx.x * outHeight +
                                                  threadIdx.y +
                                                  blockIdx.x * inSize +
                                                  blockIdx.y * stateSize];
}

template<auto Activation>
__global__ void FullyConnectBackwardErrors(num* weights, num* input, num* inErrors, num* outErrors, int stateSize,
                                           int inCount, int outCount)
{
    num accumulator = 0;
    
    for (int i = 0; i < outCount; i++)
    {
        accumulator += outErrors[i + blockIdx.x * stateSize] * weights[i + threadIdx.x * outCount];
    }
    
    inErrors[threadIdx.x + blockIdx.x * stateSize] += accumulator * Activation(input[threadIdx.x + blockIdx.x * stateSize]);
}

__global__ void FullyConnectBackwardBiases(num* outErrors, num* biasDeltas, int stateSize, int networkSize, int count)
{
    biasDeltas[threadIdx.x + blockIdx.x * networkSize] += outErrors[threadIdx.x + blockIdx.x * stateSize];
}

__global__ void FullyConnectBackwardWeights(num* input, num* outErrors, num* weightDeltas, int stateSize, int networkSize,
                                            int inSize, int outSize)
{
    num a = input[threadIdx.x + 
                  blockIdx.y * stateSize];
    
    num b = outErrors[blockIdx.x + 
                      blockIdx.y * stateSize];
    
    weightDeltas[threadIdx.x +
                 blockIdx.x * blockDim.x +
                 blockIdx.y * networkSize] += a * b;
}


// template<auto Activation>
// __global__ void FullyConnectForward(num* weights, num* biases, num* input, num* output, int stateSize,
//                                     int inCount, int outCount)
// {
//     num accumulator = 0;

//     for (int i = 0; i < inCount; i++)
//     {
//         accumulator += input[i + stateSize * blockIdx.x] * weights[threadIdx.x + i * blockDim.x];
//     }

//     num result = Activation(accumulator + biases[threadIdx.x]);

//     output[threadIdx.x + stateSize * blockIdx.x] = result;
// }


//void test()
//{
//for (in : inputs)
//{
//for (out : outputs)
//{
//for (ex : errors[out].x)
//{
//for (ey : errors[out].y)
//{
//for (wx : weights[in][out].x)
//{
//for (wy : weights[in][out].y)
//{
//errors[in][ex + wx][ey + wy] += errors[out][ex][ey] * weights[in][out][wx][wy];
//}
//}
//}
//}
//}
//}
//}

//~ Misc kernels

__global__ void ApplyDelta(num* network, num* deltas, num coefficient, int networkSize, int batchSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (id > networkSize)
    {
        return;
    }
    
    num delta = 0;
    
    for (int i = 0; i < batchSize; i++)
    {
        delta += deltas[id + networkSize * i];
    }
    
    deltas[id] = delta;
    network[id] += delta * coefficient;
}



//~ Network activation functions

void NetworkForwardPropagation(Network* network, NetworkInstance* instance, NetworkState* state, Dataset* dataset, int datasetOffset, int batchSize)
{
    static num buffer[128 * 128];
    
    Layer* layer = nullptr;
    Layer* previous = nullptr;
    
    for (int i = 0; i < network->layerCount; i++)
    {
        layer = &network->layers[i];
        
        switch (layer->type)
        {
            case LayerTypeInput2D:
            {
                int width = layer->elementWidth - layer->input2DPadding;
                int height = layer->elementHeight - layer->input2DPadding;
                
                // Reset the buffer, filling in any eventual padding
                int size = layer->elementWidth * layer->elementHeight;
                memset(buffer, 0, size * sizeof(Pixel)); 
                
                int padStart = layer->input2DPadding / 2;
                
                for (int j = 0; j < batchSize; j++)
                {
                    Pixel* in = dataset->images + (datasetOffset + j) * width * height;
                    
                    for (int x = 0; x < width; x++)
                    {
                        for (int y = 0; y < height; y++)
                        {
                            buffer[(x + padStart) * (height + layer->input2DPadding) + padStart + y] = (num)in[x * height + y] * (num)(1.0 / 255.0);
                        }
                    }
                    
                    cudaMemcpy(state->deviceMemory + j * network->stateSize + layer->stateOffset, buffer, size * sizeof(num), cudaMemcpyHostToDevice);
                }
                
                CUDA_CHECK("Train Input2D");
            }
            break;
            
            case LayerTypeConvolution2D:
            {
                dim3 dimBlock;
                dimBlock.x = layer->elementWidth;
                dimBlock.y = layer->elementHeight;
                dimBlock.z = 1;
                
                dim3 dimGrid;
                dimGrid.x = layer->elementCount;
                dimGrid.y = batchSize;
                dimGrid.z = 1;
                
                Convolve2DForward<ReLU>
                    <<<dimGrid, dimBlock>>>(instance->deviceMemory + layer->networkOffset,
                                            instance->deviceMemory + layer->networkOffset + layer->conv2DBiasOffset,
                                            state->deviceMemory + previous->stateOffset,
                                            state->deviceMemory + layer->stateOffset,
                                            network->stateSize,
                                            previous->elementWidth, previous->elementHeight, previous->elementCount,
                                            layer->elementWidth, layer->elementHeight, layer->elementCount,
                                            layer->conv2DKernelWidth);
                
                CUDA_CHECK("Train Convolution2D Forward");
            }
            break;
            
            case LayerTypePooling2D:
            {
                dim3 dimBlock;
                dimBlock.x = layer->elementWidth;
                dimBlock.y = layer->elementHeight;
                dimBlock.z = 1;
                
                dim3 dimGrid;
                dimGrid.x = layer->elementCount;
                dimGrid.y = batchSize;
                dimGrid.z = 1;
                
                Pool2DForwardMax
                    <<<dimGrid, dimBlock>>>(state->deviceMemory + previous->stateOffset,
                                            state->deviceMemory + layer->stateOffset,
                                            network->stateSize,
                                            layer->elementCount,
                                            previous->elementWidth, previous->elementHeight,
                                            layer->elementWidth, layer->elementHeight,
                                            layer->pool2DEyeWidth);
                
                CUDA_CHECK("Train Pooling2D Forward");
            }
            break;
            
            case LayerTypeFullyConnected:
            {
                dim3 dimBlock;
                dimBlock.x = layer->elementCount;
                dimBlock.y = 1;
                dimBlock.z = 1;
                
                dim3 dimGrid;
                dimGrid.x = batchSize;
                dimGrid.y = 1;
                dimGrid.z = 1;
                
                FullyConnectForward<ReLU>
                    <<<dimGrid, dimBlock>>>(instance->deviceMemory + layer->networkOffset,
                                            instance->deviceMemory + layer->networkOffset + layer->fcBiasOffset,
                                            state->deviceMemory + previous->stateOffset,
                                            state->deviceMemory + layer->stateOffset,
                                            network->stateSize,
                                            previous->elementWidth * previous->elementHeight * previous->elementCount,
                                            layer->elementCount);
                
                CUDA_CHECK("Train FC Forward");
            }
            break;
        }
        
        //cudaMemcpy(buffer, state->deviceMemory + layer->stateOffset, layer->elementWidth * layer->elementHeight * sizeof(num), cudaMemcpyDeviceToHost);
        
        //PrintImageFloating(buffer, layer->elementWidth, 1);
        
        previous = layer;
    }
}



//~ Training functions

void NetworkTrain(Network* network, NetworkInstance* instance, Dataset* dataset, int desiredBatchSize, num alpha)
{
    // Create a state capable of storing all simultaneous network features for one batch
    NetworkState state;
    NetworkStateCreate(network, &state, desiredBatchSize);
    
    // Create a corresponding state used for calculation of error
    NetworkState errors;
    NetworkStateCreate(network, &errors, desiredBatchSize);
    
    // Create a delta vector for the network
    num* deltas;
    cudaMalloc(&deltas, network->networkSize * desiredBatchSize * sizeof(num));
    CUDA_CHECK("Train Deltas Allocation");
    
    Label* correctLabels;
    cudaMalloc(&correctLabels, desiredBatchSize * sizeof(int));
    CUDA_CHECK("Train Label Allocation");
    
    // The number of batches
    // If it does not add up evenly, the last batch will be smaller
    int batches = (dataset->count - 1) / desiredBatchSize + 1; 
    
    // Working buffer used for padding inputs
    num* buffer;
    cudaMallocHost(&buffer, network->layers[0].stateSize * sizeof(num));
    CUDA_CHECK("Train Buffer Allocation");
    
    // Main training loop
    for (int batch = 0; batch < batches; batch++)
    {
        // Top level parameters for this batch
        int datasetOffset = batch * desiredBatchSize;
        int batchSize = MIN(desiredBatchSize, dataset->count - datasetOffset);
        
        // Clear the state and error arrays
        cudaMemset(state.deviceMemory, 0, network->stateSize * batchSize * sizeof(num));
        cudaMemset(errors.deviceMemory, 0, network->stateSize * batchSize * sizeof(num));
        CUDA_CHECK("Train States Memset");
        
        
        
        //- Forward propagation
        
        //auto start = std::chrono::steady_clock::now();
        
        NetworkForwardPropagation(network, instance, &state, dataset, datasetOffset, batchSize);
        
        //auto end = std::chrono::steady_clock::now();
        //PRINT_INFO("Time taken: %f", std::chrono::duration<double>(end - start).count());
        
        
        
        //- Validation
        
        // Clear the delta array
        cudaMemset(deltas, 0, network->networkSize * batchSize * sizeof(num));
        CUDA_CHECK("Train Deltas Memset");
        
        cudaMemcpy(correctLabels, dataset->labels + datasetOffset, batchSize * sizeof(Label), cudaMemcpyHostToDevice);
        CUDA_CHECK("Train Softmax Copy");
        
        // Seed the error state with softmax errors in the output layer corresponding with the correct label
        Layer* outputLayer = &network->layers[network->layerCount - 1];
        
        SoftmaxComputeErrors<<<batchSize, outputLayer->elementCount>>>(state.deviceMemory + outputLayer->stateOffset,
                                                                       errors.deviceMemory + outputLayer->stateOffset,
                                                                       correctLabels, 
                                                                       network->stateSize, 
                                                                       outputLayer->elementCount);
        
        CUDA_CHECK("Train Softmax");
        
        num outs[10];
        num errs[10];
        
        cudaMemcpy(outs, state.deviceMemory + outputLayer->stateOffset, 10 * sizeof(num), cudaMemcpyDeviceToHost);
        cudaMemcpy(errs, errors.deviceMemory + outputLayer->stateOffset, 10 * sizeof(num), cudaMemcpyDeviceToHost);
        
        
        
        //- Backpropagation
        
        Layer* layer = nullptr;
        Layer* previous = nullptr;
        
        for (int i = network->layerCount - 1; i >= 1 ; i--)
        {
            layer = &network->layers[i];
            previous = &network->layers[i - 1];
            
            bool backpropagateErrors = i > 1;
            
            switch (layer->type)
            {
                case LayerTypeInput2D:
                {
                    // Shouldn't be hit
                    PRINT_ERROR("Hit input layer in backpropagation.");
                }
                break;
                
                case LayerTypeConvolution2D:
                {
                    dim3 dimBlock;
                    dim3 dimGrid;
                    
                    dimBlock.x = layer->elementWidth;
                    dimBlock.y = layer->elementHeight;
                    dimBlock.z = 1;
                    
                    dimGrid.x = layer->elementCount;
                    dimGrid.y = batchSize;
                    dimGrid.z = 1;
                    
                    Convolve2DBackwardSum
                        <<<dimGrid, dimBlock>>>(instance->deviceMemory + layer->networkOffset,
                                                errors.deviceMemory + previous->stateOffset,
                                                errors.deviceMemory + layer->stateOffset,
                                                network->stateSize,
                                                previous->elementWidth, previous->elementHeight, previous->elementCount,
                                                layer->elementWidth, layer->elementHeight, layer->elementCount,
                                                layer->conv2DKernelWidth);
                    
                    CUDA_CHECK("Train Convolution2D Backward Sum");
                    
                    dimBlock.x = previous->elementWidth;
                    dimBlock.y = previous->elementHeight;
                    dimBlock.z = 1;
                    
                    dimGrid.x = previous->elementCount;
                    dimGrid.y = batchSize;
                    dimGrid.z = 1;
                    
                    BackwardActivation<ReLUDeriv>
                        <<<dimGrid, dimBlock>>>(state.deviceMemory + previous->stateOffset,
                                                errors.deviceMemory + previous->stateOffset,
                                                network->stateSize, 
                                                previous->elementWidth * previous->elementHeight);
                    
                    CUDA_CHECK("Train Convolution2D Backward Activation");
                    
                    dimBlock.x = layer->elementWidth;
                    dimBlock.y = 1;
                    dimBlock.z = 1;
                    
                    dimGrid.x = layer->elementCount;
                    dimGrid.y = batchSize;
                    dimGrid.z = 1;
                    
                    Convolve2DBackwardBiases
                        <<<dimGrid, dimBlock>>>(errors.deviceMemory + layer->stateOffset,
                                                deltas + layer->networkOffset + layer->conv2DBiasOffset,
                                                network->stateSize, network->networkSize,
                                                layer->elementWidth, layer->elementHeight);
                    
                    CUDA_CHECK("Train Convolution2D Backward Biases");
                    
                    dimBlock.x = layer->conv2DKernelWidth;
                    dimBlock.y = layer->conv2DKernelWidth;
                    dimBlock.z = 1;
                    
                    dimGrid.x = layer->elementCount;
                    dimGrid.y = batchSize;
                    dimGrid.z = 1;
                    
                    Convolve2DBackwardWeights
                        <<<dimGrid, dimBlock>>>(state.deviceMemory + previous->stateOffset,
                                                errors.deviceMemory + layer->stateOffset,
                                                deltas + layer->networkOffset,
                                                network->stateSize, network->networkSize,
                                                previous->elementWidth, previous->elementHeight, previous->elementCount,
                                                layer->elementWidth, layer->elementHeight, layer->elementCount,
                                                layer->conv2DKernelWidth);
                    
                    CUDA_CHECK("Train Convolution2D Backward Weights");
                }
                break;
                
                case LayerTypePooling2D:
                {
                    dim3 dimBlock;
                    dimBlock.x = layer->elementWidth;
                    dimBlock.y = layer->elementHeight;
                    dimBlock.z = 1;
                    
                    dim3 dimGrid;
                    dimGrid.x = layer->elementCount;
                    dimGrid.y = batchSize;
                    dimGrid.z = 1;
                    
                    Pool2DBackwardMax
                        <<<dimGrid, dimBlock>>>(state.deviceMemory + previous->stateOffset,
                                                errors.deviceMemory + previous->stateOffset,
                                                errors.deviceMemory + layer->stateOffset,
                                                network->stateSize,
                                                layer->elementCount,
                                                previous->elementWidth, previous->elementHeight,
                                                layer->elementWidth, layer->elementHeight,
                                                layer->pool2DEyeWidth);
                    
                    CUDA_CHECK("Train Pooling2D Backward");
                }
                break;
                
                case LayerTypeFullyConnected:
                {
                    dim3 dimBlock;
                    dim3 dimGrid;
                    
                    if (backpropagateErrors)
                    {
                        dimBlock.x = previous->elementCount * previous->elementWidth * previous->elementHeight;
                        dimBlock.y = 1;
                        dimBlock.z = 1;
                        
                        dimGrid.x = batchSize;
                        dimGrid.y = 1;
                        dimGrid.z = 1;
                        
                        FullyConnectBackwardErrors<ReLUDeriv>
                            <<<dimGrid, dimBlock>>>(instance->deviceMemory + layer->networkOffset,
                                                    state.deviceMemory + previous->stateOffset,
                                                    errors.deviceMemory + previous->stateOffset,
                                                    errors.deviceMemory + layer->stateOffset,
                                                    network->stateSize,
                                                    previous->elementWidth * previous->elementHeight * previous->elementCount,
                                                    layer->elementCount);
                        
                        CUDA_CHECK("Train FC Backward Sum and Activation");
                    }
                    
                    dimBlock.x = layer->elementCount;
                    dimBlock.y = 1;
                    dimBlock.z = 1;
                    
                    dimGrid.x = batchSize;
                    dimGrid.y = 1;
                    dimGrid.z = 1;
                    
                    FullyConnectBackwardBiases
                        <<<dimGrid, dimBlock>>>(errors.deviceMemory + layer->stateOffset,
                                                deltas + layer->networkOffset + layer->fcBiasOffset,
                                                network->stateSize, network->networkSize,
                                                layer->elementCount);
                    
                    CUDA_CHECK("Train FC Backward Biases");
                    
                    dimBlock.x = previous->elementWidth * previous->elementHeight * previous->elementCount;
                    dimBlock.y = 1;
                    dimBlock.z = 1;
                    
                    dimGrid.x = layer->elementCount;
                    dimGrid.y = batchSize;
                    dimGrid.z = 1;
                    
                    FullyConnectBackwardWeights
                        <<<dimGrid, dimBlock>>>(state.deviceMemory + previous->stateOffset,
                                                errors.deviceMemory + layer->stateOffset,
                                                deltas + layer->networkOffset,
                                                network->stateSize, network->networkSize,
                                                previous->elementWidth * previous->elementHeight * previous->elementCount,
                                                layer->elementCount);
                    
                    CUDA_CHECK("Train FC Backward Weights");
                }
                break;
            }
        }
        
        int applyBlockSize = 512;
        int applyBlockCount = (network->networkSize - 1) / 512 + 1;
        
        cudaDeviceSynchronize();
        
        if (0) //batch == 0)
        {
            // num temp[16800];
            
            // cudaMemcpy(temp, deltas, sizeof(temp), cudaMemcpyDeviceToHost);
            // for (int t = 0; t < 16800; t++) { printf("%.3f ", temp[t]); } printf("\n\n");
            
            num temp[442];
            
            cudaMemcpy(temp, deltas + network->layers[2].networkOffset, sizeof(temp), cudaMemcpyDeviceToHost);
            for (int t = 0; t < 442; t++) { printf("%.3f ", temp[t]); } printf("\n\n");
            
            cudaMemcpy(temp, instance->deviceMemory + network->layers[2].networkOffset, sizeof(temp), cudaMemcpyDeviceToHost);
            for (int t = 0; t < 442; t++) { printf("%.3f ", temp[t]); } printf("\n\n");
        }
        
        ApplyDelta<<<applyBlockCount, applyBlockSize>>>(instance->deviceMemory, 
                                                        deltas, 
                                                        alpha / batchSize, 
                                                        network->networkSize, 
                                                        batchSize);
        CUDA_CHECK("Apply Weights");
        
        cudaDeviceSynchronize();
        
        if (0) //batch == 0)
        {
            // num temp[16800];
            
            // cudaMemcpy(temp, deltas, sizeof(temp), cudaMemcpyDeviceToHost);
            // for (int t = 0; t < 16800; t++) { printf("%.3f ", temp[t]); } printf("\n\n");
            
            printf("-\n\n");
            
            num temp[442];
            
            cudaMemcpy(temp, deltas + network->layers[2].networkOffset, sizeof(temp), cudaMemcpyDeviceToHost);
            for (int t = 0; t < 442; t++) { printf("%.3f ", temp[t]); } printf("\n\n");
            
            cudaMemcpy(temp, instance->deviceMemory + network->layers[2].networkOffset, sizeof(temp), cudaMemcpyDeviceToHost);
            for (int t = 0; t < 442; t++) { printf("%.3f ", temp[t]); } printf("\n\n");
        }
    }
    
    // Deinitialize
    cudaFreeHost(buffer);
    cudaFree(deltas);
    cudaFree(correctLabels);
    CUDA_CHECK("Train Free");
    
    NetworkStateDestroy(&state);
    NetworkStateDestroy(&errors);
}



//~ Testing

void NetworkTest(Network* network, NetworkInstance* instance, Dataset* dataset)
{
    int desiredBatchSize = 1;
    
    NetworkState state;
    NetworkStateCreate(network, &state, desiredBatchSize);
    
    // The number of batches
    // If it does not add up evenly, the last batch will be smaller
    int batches = (dataset->count - 1) / desiredBatchSize + 1; 
    
    int confusionMatrix[10][10] = { 0 };
    
    int correct = 0;
    int total = dataset->count;
    
    for (int batch = 0; batch < batches; batch++)
    {
        // Top level parameters for this batch
        int datasetOffset = batch * desiredBatchSize;
        int batchSize = MIN(desiredBatchSize, dataset->count - datasetOffset);
        
        // Clear the state and error arrays
        cudaMemset(state.deviceMemory, 0, network->stateSize * batchSize * sizeof(num));
        CUDA_CHECK("Train States Memset");
        
        NetworkForwardPropagation(network, instance, &state, dataset, datasetOffset, batchSize);
        
        // Seed the error state with softmax errors in the output layer corresponding with the correct label
        Layer* outputLayer = &network->layers[network->layerCount - 1];
        
        num outs[10];
        cudaMemcpy(outs, state.deviceMemory + outputLayer->stateOffset, 10 * sizeof(num), cudaMemcpyDeviceToHost);
        
        int max = 0;
        for (int i = 1; i < 10; i++)
        {
            if (outs[i] > outs[max])
            {
                max = i;
            }
        }
        
        int label = (int)dataset->labels[batch];
        
        correct += label == max;
        
        confusionMatrix[label][max] += 1;
    }
    
    PrintResult(confusionMatrix);
    
    printf("\nCorrect guesses: %i / %i (%.4f%%)\n", correct, total, (double)correct / total * 100);
    
    NetworkStateDestroy(&state);
}



//~ Entry point

int main()
{
    // Tweakable parameters
    const int batchSize = 300;
    const num alpha = (num)0.5;
    
    // Import data
    Dataset datasetTrain;
    Dataset datasetTest;
    ReadDataset(&datasetTrain, COUNT_TRAIN, IMAGE_WIDTH, IMAGE_WIDTH, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL);
    ReadDataset(&datasetTest,  COUNT_TEST,  IMAGE_WIDTH, IMAGE_WIDTH, FILE_TEST_IMAGE,  FILE_TEST_LABEL);
    
    // Create the network structure
    Network network = {};
    //NetworkCreateLeNet5Reference(&network);
    //NetworkCreateSimple(&network);
    NetworkCreateNNADL(&network);
    
    // Create an instance of the network
    NetworkInstance instance;
    NetworkInstanceCreate(&network, &instance, true);
    
    // Train the network
    //NetworkTrain(&network, &instance, &datasetTrain, batchSize, alpha);
    
    for (int i = 0; i < 50; i++)
    {
        NetworkTrain(&network, &instance, &datasetTrain, 20, 0.5);
        printf(".");
        fflush(stdout);
    }
    printf("\n");
    
    
    // Test the trained network
    NetworkTest(&network, &instance, &datasetTest);
    
    // Deinitialize
    NetworkInstanceDestroy(&instance);
    NetworkDestroy(&network);
    return 0;
}