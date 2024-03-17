#pragma once

// Sai: In LeNet, a 5x5 convolution kernel (or mask) is used
#define LENGTH_KERNEL	5

#define LENGTH_FEATURE0	32 // Layer 0's input image dimension: 32 x 32
#define LENGTH_FEATURE1	(LENGTH_FEATURE0 - LENGTH_KERNEL + 1) // Layer 1's image dimension: 28 x 28
#define LENGTH_FEATURE2	(LENGTH_FEATURE1 >> 1) // Layer 2's image dimension: 14 x 14
#define LENGTH_FEATURE3	(LENGTH_FEATURE2 - LENGTH_KERNEL + 1) // Layer 3's image dimension: 10 x 10
#define	LENGTH_FEATURE4	(LENGTH_FEATURE3 >> 1) // Layer 4's image dimension: 5 x 5
#define LENGTH_FEATURE5	(LENGTH_FEATURE4 - LENGTH_KERNEL + 1) // Layer 5

// Sai: Check the LeNet architecture diagram (Figure 16.2 of textbook)
// to understand what the following numbers represent  
#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define OUTPUT          10

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];


typedef struct LeNet5
{
	double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
	double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];

	double bias0_1[LAYER1];
	double bias2_3[LAYER3];
	double bias4_5[LAYER5];
	double bias5_6[OUTPUT];

} LeNet5;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
	double output[OUTPUT];

} Feature;

void TrainBatch(LeNet5* lenet, image* inputs, uint8* labels, int batchSize);

void Train(LeNet5* lenet, image input, uint8 label);

uint8 Predict(LeNet5* lenet, image input, uint8 count);

void Initial(LeNet5* lenet);

void DebugForward(LeNet5* lenet, Feature* features, image input);
void DebugBackward(LeNet5* lenet, Feature* features, Feature* errors, LeNet5* deltas, uint8 label);
void DebugApply(LeNet5* lenet, LeNet5* deltas);

