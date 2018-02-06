#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "/Users/oncio/Documents/Visual Studio 2015/Projects/LabPPDCuda/LabPPDCuda/checkErrors.h"


using namespace std;
using namespace cv;

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }



__global__ void rgba_to_greyscale_kernel(const uchar4* const rgbaImage,
	unsigned char* const greyImage,
	int numRows, int numCols)
{
	const long pointIndex = threadIdx.x + blockDim.x*blockIdx.x;

	if (pointIndex<numRows*numCols) { // this is necessary only if too many threads are started
		uchar4 const imagePoint = rgbaImage[pointIndex];
		greyImage[pointIndex] = .299f*imagePoint.x + .587f*imagePoint.y + .114f*imagePoint.z;
	};

}

void your_rgba_to_greyscale(uchar4 * const d_rgbaImage,
	unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	const int blockThreadSize = 256;
	const int numberOfBlocks = 1 + ((numRows*numCols - 1) / blockThreadSize); // a/b rounded up
	const dim3 blockSize(blockThreadSize, 1, 1);
	const dim3 gridSize(numberOfBlocks, 1, 1);
	rgba_to_greyscale_kernel <<<gridSize, blockSize >>>(d_rgbaImage, d_greyImage, numRows, numCols);
/*
	void* args[] = { d_rgbaImage, d_greyImage, &numRows, &numCols };
	cudaLaunchKernel(
		(const void*)&rgba_to_greyscale_kernel, // pointer to kernel func.
		dim3(gridSize), // grid
		dim3(blockSize),
		args);
		*/
}




void preProcess(uchar4 **inputImage, unsigned char **greyImage,
	uchar4 **d_rgbaImage, unsigned char **d_greyImage,
	const std::string &filename) {
	checkCudaErrors(cudaFree(0));

	cv::Mat image;
	image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty()) {
		std::cerr << "Couldn't open file: " << filename << std::endl;
		exit(1);
	}

	cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

	imageGrey.create(image.rows, image.cols, CV_8UC1);

		if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
		std::cerr << "Images aren't continuous!! Exiting." << std::endl;
		exit(1);
	}

	*inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
	*greyImage = imageGrey.ptr<unsigned char>(0);

	const size_t numPixels = numRows() * numCols();
	//allocate memory on the device for both input and output
	checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
	checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
	checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

																					 //copy input array to the GPU
	checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

	d_rgbaImage__ = *d_rgbaImage;
	d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
	const int numPixels = numRows() * numCols();
	//copy the output back to the host
	checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	//output the image
	cv::imwrite(output_file.c_str(), imageGrey);

	//cleanup
	cudaFree(d_rgbaImage__);
	cudaFree(d_greyImage__);
}

int main(int argc, char **argv) {
	uchar4        *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;
	std::string input_file;
	std::string output_file;
	if (argc == 3) {
		input_file = std::string(argv[1]);
		output_file = std::string(argv[2]);
	}
	else {
		std::cerr << "No args were sent" << std::endl;
		exit(1);
	}
	//load the image and give us our input and output pointers
	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	//GpuTimer timer;
	//timer.Start();
	//call the grayscale code

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	your_rgba_to_greyscale(d_rgbaImage, d_greyImage, numRows(), numCols());
	//timer.Stop();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	printf("\n");
	//int err = printf("%f msecs.\n", timer.Elapsed());
	printf("The operation took %f msecs.\n", milliseconds);

	system("pause");

	//check results and output the grey image
	postProcess(output_file);

	return 0;
}

