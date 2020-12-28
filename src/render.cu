
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include "memory"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func
              << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}
// __device__ float hit_sphere(const point3& center, double radius, const ray& r) {
//   vec3 oc = r.origin() - center;
//   auto a = r.direction().length_squared();
//   auto half_b = dot(oc, r.direction());
//   auto c = oc.length_squared() - radius * radius;
//   auto discriminant = half_b * half_b - a * c;
//   if (discriminant < 0) {
//     return -1.0;
//   } else {
//     return (-half_b - sqrt(discriminant)) / a;
//   }
// }

__device__ vec3 ray_color(const ray& r, hitable** world) {
  hit_record rec;

  if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
    return 0.5 * (rec.normal + vec3(1, 1, 1));
  }
  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

__global__ void render(
    vec3* const f,
    const unsigned max_x,
    const unsigned max_y,
    const unsigned ns,
    camera** cam,
    hitable** world,
    curandState* rand_state) {
  const unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
  size_t idx = y * max_x + x;
  if (x >= max_x || y >= max_y)
    return;

  curand_init(1984, 0, 0, rand_state + idx);
  curandState local_rand_state = rand_state[idx];
  vec3 color{0, 0, 0};

  for (unsigned i=0; i < ns; ++i) {
    float u = (x + curand_uniform(&local_rand_state)) / float(max_x);
    float v = (y + curand_uniform(&local_rand_state)) / float(max_y);

    auto r = (*cam)->get_ray(u, v);

    color += ray_color(r, world);
  }
  f[idx] = color / float(ns);
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
    *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
    *d_world = new hitable_list(d_list, 2);
    *d_cam = new camera();
  }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
    delete *cam;
  }
}

void render_image(float* const h_fb, const unsigned nx, const unsigned ny) {
  vec3* d_fb;

  const unsigned tx = 8, ty = 8, num_pixels = nx * ny;
  const unsigned samples_per_pixel = 100;
  const size_t fb_size = num_pixels * sizeof(vec3);

  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
  checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  camera** cam;
  checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

  hitable** d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
  hitable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
  create_world<<<1, 1>>>(d_list, d_world, cam);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  render<<<blocks, threads>>>(d_fb, nx, ny, samples_per_pixel, cam, d_world, d_rand_state);

  free_world<<<1, 1>>>(d_list, d_world, cam);

  checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(d_fb));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(cam));
}
