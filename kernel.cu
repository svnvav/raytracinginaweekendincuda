#include <iostream>
#include <fstream>
#include "vec3.h"
#include "device_launch_parameters.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(vec3 *fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	fb[pixel_index] = vec3(float(i) / max_x, float(j) / max_y, 0.2);
}

int main()
{
	int nx = 1024;
	int ny = 768;

	int tx = 8;
	int ty = 8;

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads>>> (fb, nx, ny);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	std::ofstream outfile;
	outfile.open("HelloWorld.ppm");
	outfile << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			int ir = int(255.99 * fb[pixel_index][0]);
			int ig = int(255.99 * fb[pixel_index][1]);
			int ib = int(255.99 * fb[pixel_index][2]);
			outfile << ir << " " << ig << " " << ib << "\n";
		}
	}
	checkCudaErrors(cudaFree(fb));

    return 0;
}


