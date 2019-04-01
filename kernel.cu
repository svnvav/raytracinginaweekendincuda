#include <iostream>
#include <fstream>
#include "ray.h"
#include "time.h"
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

__device__ vec3 color(const ray& r) {
	vec3 unit_direction = unit_vector(r.direction());
	float t = 0.5f*(unit_direction.y() + 1.0f);
	return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y,
	vec3 origin, vec3 lower_left_corner, vec3 horizontal, vec3 vertical) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;

	float u = float(i) / float(max_x);
	float v = float(j) / float(max_y);
	ray r(origin, lower_left_corner + u * horizontal + v * vertical);
	fb[pixel_index] = color(r);
}

int main()
{
	int nx = 1024;
	int ny = 512;
	vec3 origin(0 ,0, 0);
	vec3 lower_left_corner(-2, -1, -1);
	vec3 horizontal(4, 0, 0);
	vec3 vertical(0, 2, 0);

	int tx = 8;
	int ty = 8;

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(vec3);

	vec3 *fb;

	clock_t start, stop;
	start = clock();

	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Start and allocation time spent: " << timer_seconds << " seconds.\n";

	start = clock();

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads>>> (fb, nx, ny, origin, lower_left_corner, horizontal, vertical);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Kernel work and sync time spent: " << timer_seconds << " seconds.\n";

	start = clock();

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

	stop = clock();
	timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cout << "Output time spent: " << timer_seconds << " seconds.\n";

	checkCudaErrors(cudaFree(fb));

    return 0;
}


