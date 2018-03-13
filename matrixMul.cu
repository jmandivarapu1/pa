/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
* image: Image to be blurred
* i: Pixel position y
* j: Pixel position x
* return average channel vector
**/

__device__
void get_average(const int &y, const int &x, int *image, int *result, const int &height, const int &width, const int &kernel){
    int blue, green, red, dy, dx, cont, limit;
    blue = green = red = cont = 0;
    
    // Limit of adjacent positions
    limit = kernel / 2;

    // Traverse all adjacents positions
    for( int i = -limit; i <= limit; ++i ){
        for( int j = -limit; j <= limit; ++j ){
            dy = y + i;
            dx = x + j;

            // Verify if point is inside of image
            if( 0 <= dy && dy < height && 0 <= dx && dx < width ){
                blue += image[dy * width * 3 + dx * 3 + 0];
                green += image[dy * width * 3 + dx * 3 + 1];
                red += image[dy * width * 3 + dx * 3 + 2];
                ++cont;
            }
        }
    }

    // Calculate average of each channel
    blue /= cont;
    green /= cont;
    red /= cont;
	
	result[y * width * 3 + x * 3 + 0] = blue;
	result[y * width * 3 + x * 3 + 1] = green;
	result[y * width * 3 + x * 3 + 2] = red;
}

__global__
void blur(int *image, int *result, int *height, int *width, int *kernel) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int n = *height * *width;
    // Calculate pixel average for every pixel   
    int i, j;
    for(int k = index; k < n; k += (stride++)) {
    	i = k / *width;
    	j = k % *width;
    	get_average(i, j, image, result, *height, *width, *kernel);
    }
}

int main(int argc, char** argv){

	Mat image, result,result_pyramid;
	
	int *height_d;
	int *width_d;
	int *kernel_d;
	
	int height;
	int width;
	int kernel;
	int num_threads;
	int num_blocks;

    cudaMalloc(&height_d, sizeof(int));
    cudaMalloc(&width_d, sizeof(int));
    cudaMalloc(&kernel_d, sizeof(int));
  
    if (argc != 5){
        printf("Use: ./blurSecuencial.out <img_path> <num_kernel> <num_threads> <num_blocks>\n");
        return -1;
    }

    image = imread(argv[1], 1);

    if ( !image.data ){
        printf("Image couldn't be read. \n");
        return -1;
    }

    kernel = atoi(argv[2]);
    num_threads = atoi(argv[3]);
    num_blocks = atoi(argv[4]);

    height = image.size().height;
    width = image.size().width;
	
	cudaMemcpy(kernel_d, &kernel, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(width_d, &width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(height_d, &height, sizeof(int), cudaMemcpyHostToDevice);

	int *image_mat_d;
	int *image_mat = (int *) malloc (height * width * 3 * sizeof(int));
	
	int *result_mat_d;
	int *result_mat = (int *) malloc (height * width * 3 * sizeof(int));

  	cudaMalloc(&image_mat_d, sizeof(int) * height * width * 3);
        cudaMalloc(&result_mat_d, sizeof(int) * height * width * 3);
	
    // Create result image with same dimensions
    result = Mat(height, width, CV_8UC3);
    result_pyramid=Mat(height/2,width/2,CV_8UC3);
    
    for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		for (int k = 0; k < 3; ++k) {
    			image_mat[i * width * 3 + j * 3 + k] = image.at<Vec3b>(i, j)[k];
    		}
    	}
    }

   
    cudaMemcpy(image_mat_d, image_mat, sizeof(int) * height * width * 3, cudaMemcpyHostToDevice);
    
//	int num_threads = 256;
//	int num_blocks = (height * width + num_threads - 1) / num_threads;

    blur<<<num_blocks, num_threads>>>(image_mat_d, result_mat_d, height_d, width_d, kernel_d);

    cudaMemcpy(result_mat, result_mat_d, sizeof(int) * height * width * 3, cudaMemcpyDeviceToHost);

    printf("%d--%d\n",height,width);
    for (int i = 0; i < height; ++i) {
	    for (int j = 0; j < width; ++j) {
		    for (int k = 0; k < 3; ++k) {
			       // printf("%d--%d--%d--%d\n",i,j,k,(i * (width) * 3 + j * 3 + k));
				result.at<Vec3b>(i, j)[k] = result_mat[i * (width) * 3 + j * 3 + k];
		    }
	    }
    }
   printf("%d--%d\n",height,width);

   for (int i = 0; i < height-1;++i) {
            for (int j = 0; j < width-1; ++ j) 
                 {
 // printf("%d--%d--%d--%d--%d\n",i,j,k,(i * (width) * 3 + j * 3 + k),(i * (width) * 3 + (j+1) * 3 + k));
                    printf("%d--%d\n",i,j);
                    for (int k = 0; k < 3; ++k) {
             printf("%d--%d--%d--%d--%d--%d--%d\n",i,j,k,(i/2),(j/2),(i * (width) * 3 + j * 3 + k),(i * (width) * 3 + (j+1) * 3 + k));
                            result_pyramid.at<Vec3b>((i/2), (j/2))[k] = (result_mat[i * (width) * 3 + j * 3 + k]
                                                                        +result_mat[(i) * (width) * 3 + (j+1)* 3 + (k)])/2;
                    }
            }
    }

    printf("%s","its finished the process");
    Mat3b imageF_8UC3;
    result.convertTo(imageF_8UC3, CV_8UC3, 255);
    
    waitKey(0);
    string file_name,file_name_py;
    file_name = "blurred_out.png";
    file_name_py="test_4.jpg";
    //cv ::imshow("test.jpg",result);
    imwrite(file_name, result);
    imwrite(file_name_py, result_pyramid); 
    free(image_mat);
    free(result_mat);
    return 0;
}


//nvcc promising.cu `pkg-config --cflags --libs opencv` -o blur-effect.out
//./blur-effect.out img1.jpg  9 256 652
