
#include <float.h>
#include <iostream>
#include "hitable.h"
#include "hitable_list.h"
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

__device__ bool hit_sphere(const vec3& center, const float radius, const ray& r) {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0 * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4.0 * a * c;
  return (discriminant > 0);
}

__device__ vec3 color(const ray& r, hitable** world) {
  hit_record rec;

  if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {

    return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
  }
}

__global__ void render(
    vec3* const f,
    const unsigned max_x,
    const unsigned max_y,
    const vec3 origin,
    const vec3 lower_left_corner,
    const vec3 horizontal,
    const vec3 vertical,
    hitable** world) {
  const unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned y = threadIdx.y + blockIdx.y * blockDim.y;
  size_t idx = y * max_x + x;
  if (x >= max_x || y >= max_y)
    return;
  float u = float(x) / float(max_x);
  float v = float(y) / float(max_y);

  ray r(origin, lower_left_corner + u * horizontal + v * vertical);

  f[idx] = color(r, world);
}

__global__ void create_world(hitable** d_list, hitable** d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
    *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
    *d_world = new hitable_list(d_list, 2);
  }
}

__global__ void free_world(hitable** d_list, hitable** d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    delete *(d_list);
    delete *(d_list + 1);
    delete *d_world;
  }
}

void render_image(float* const h_fb, const unsigned nx, const unsigned ny) {
  vec3* d_fb;

  const unsigned tx = 8, ty = 8, num_pixels = nx * ny;
  const size_t fb_size = num_pixels * sizeof(vec3);

  checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  const vec3 origin(0., 0., 0.), lower_left_corner(-2., -1., -1.), horizontal(4., 0., 0.), vertical(0., 2., 0.);

  hitable** d_list;
  checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hitable*)));
  hitable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
  create_world<<<1, 1>>>(d_list, d_world);
  render<<<blocks, threads>>>(d_fb, nx, ny, origin, lower_left_corner, horizontal, vertical, d_world);

  free_world<<<1, 1>>>(d_list, d_world);

  checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_fb));
}
