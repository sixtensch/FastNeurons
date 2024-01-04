
//~ Includes

#include <cstdlib>
#include <cstdio>
#include <cstdint>



//~ Training data definition

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE  "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL  "t10k-labels-idx1-ubyte"

#define COUNT_TRAIN 60000
#define COUNT_TEST  10000

#define IMAGE_WIDTH 28
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_WIDTH)



//~ System definition

#define LENGTH_KERNEL   5

#define LENGTH_FEATURE0 32 // Layer 0's input image dimension: 32 x 32
#define LENGTH_FEATURE1 (LENGTH_FEATURE0 - LENGTH_KERNEL + 1) // Layer 1's image dimension: 28 x 28
#define LENGTH_FEATURE2 (LENGTH_FEATURE1 >> 1) // Layer 2's image dimension: 14 x 14
#define LENGTH_FEATURE3 (LENGTH_FEATURE2 - LENGTH_KERNEL + 1) // Layer 3's image dimension: 10 x 10
#define LENGTH_FEATURE4 (LENGTH_FEATURE3 >> 1) // Layer 4's image dimension: 5 x 5
#define LENGTH_FEATURE5 (LENGTH_FEATURE4 - LENGTH_KERNEL + 1) // Layer 5

#define INPUT  1
#define LAYER1 6
#define LAYER2 6
#define LAYER3 16
#define LAYER4 16
#define LAYER5 120
#define OUTPUT 10

#define PADDING 2



//~ Macro definitions

// Column major get
#define GET(data, x, y, width) (data[x * width + y])



//~ Data typedefs

typedef float          num;
typedef unsigned char  byte;

typedef byte           Image[28 * 28];
typedef unsigned char  Label;



//~ Struct definitions

struct Dataset
{
    int count;
    Image* images;
    Label* labels;
};



//~ Activation functions

num ReLU(num x)
{
    return x * (x > 0);
}

num ReLUDeriv(num y)
{
    return y > 0;
}

#define PRELU_A ((num)0.02)
num PReLU(num x)
{
    bool p = x > 0;
    return 
        x * p + 
        x * PRELU_A * !p;
}

num PReLUDeriv(num y)
{
    bool p = y > 0;
    return p + !p * PRELU_A;
}



//~ Kernels

__global__ void Convolve()
{
}



//~ Help functions

int ReadDataset(Dataset* dataset, int count, const char* pathImages, const char* pathLabels)
{
    FILE* fileImage = fopen(pathImages, "rb");
    FILE* fileLabel = fopen(pathLabels, "rb");
    
    if (fileImage == nullptr || fileLabel == nullptr)
    {
        return 1;
    }
    
    dataset->images = (Image*)malloc(sizeof(Image) * count);
    dataset->labels = (Label*)malloc(sizeof(Label) * count);
    
	fseek(fileImage, 16, SEEK_SET);
	fseek(fileLabel, 8, SEEK_SET);
    
	fread(dataset->images, sizeof(Image) * count, 1, fileImage);
	fread(dataset->labels, sizeof(Label) * count, 1, fileLabel);
    
	fclose(fileImage);
	fclose(fileLabel);
    
    return 0;
}

// Provided with the example
void PrintResult(int confusion_matrix[OUTPUT][OUTPUT])
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


void PrintImage(byte* pixels, int width, int sampling = 2, bool tendToMax = true, bool drawFrame = true)
{
    static const char characters[] = " .*#";
    static const byte characterCount = sizeof(characters) - 1;
    
    static char buffer[(32 + 3) * (32 + 2) + 1] = { '\0' };
    int i = 0;
    
#define BPRINT(c) buffer[i] = c; i++;
    
    if (drawFrame)
    {
        BPRINT('*');
        for (int f = 0; f < width / sampling; f++) 
        {
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
                    
                    byte current = GET(pixels, x, y, width);
                    if (current > max)
                    {
                        max = current;
                    }
                }
            }
            
            byte value = 0;
            unsigned int average = sum / (sampling * sampling);
            
            if (tendToMax)
            {
                // Middle point between average and max, gives good legibility
                value = (byte)((average + max) / 2);
            }
            else
            {
                value = (byte)average;
            }
            
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
        }
        BPRINT('*');
        BPRINT('\n');
    }
    
    buffer[i] = '\0';
    
#undef BPRINT
    
    printf("%s", buffer);
}



//~ Entry point

int main()
{
    // Import data
    
    Dataset datasetTrain;
    Dataset datasetTest;
    ReadDataset(&datasetTrain, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL);
    ReadDataset(&datasetTest, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL);
    
    for (int i = 0; i < 50; i++)
    {
        PrintImage(datasetTrain.images[i], 28, 1);
    }
    
    return 0;
}