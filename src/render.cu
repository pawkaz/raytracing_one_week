
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include "camera.h"
#include "hitable.h"
#include "hitable_list.h"
#include "material.h"
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

__device__ vec3 ray_color(const ray& r, hitable** world, curandState* r_state) {
  ray c_ray = r;
  color c_attenuation(1, 1, 1);

  for (unsigned step = 0; step < 50; ++step) {
    hit_record rec;
    if ((*world)->hit(c_ray, 0.001f, FLT_MAX, rec)) {
      ray scattered;
      vec3 attenuation;
      if (rec.mat_ptr->scatter(c_ray, rec, attenuation, scattered, r_state)) {
        c_attenuation *= attenuation;
        c_ray = scattered;
      } else {
        return vec3(0.0, 0.0, 0.0);
      }
    } else {
      vec3 unit_direction = unit_vector(c_ray.direction());
      auto t = 0.5f * (unit_direction.y() + 1.0f);
      auto c = (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
      return c_attenuation * c;
    }
  }
  return vec3(0.0, 0.0, 0.0);
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

  curand_init(1984 + idx, 0, 0, &rand_state[idx]);
  curandState local_rand_state = rand_state[idx];
  vec3 color{0, 0, 0};

  for (unsigned i = 0; i < ns; ++i) {
    float u = (x + curand_uniform(&local_rand_state)) / float(max_x);
    float v = (y + curand_uniform(&local_rand_state)) / float(max_y);

    auto r = (*cam)->get_ray(u, v, &local_rand_state);

    color += ray_color(r, world, &local_rand_state);
  }
  color /= float(ns);
  color = vec3(sqrt(color.r()), sqrt(color.g()), sqrt(color.b()));
  f[idx] = color;
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // auto list = new sphere[2];
    curandState c_state;
    curand_init(10000, 0, 0, &c_state);

    d_list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
    d_list[1] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5f));
    d_list[2] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    d_list[3] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0));
    auto idx =  4;

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = curand_uniform(&c_state);
            point3 center(a + 0.9*curand_uniform(&c_state), 0.2, b + 0.9*curand_uniform(&c_state));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material* sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto a_ = vec3(curand_uniform(&c_state), curand_uniform(&c_state), curand_uniform(&c_state));
                    auto b_ = vec3(curand_uniform(&c_state), curand_uniform(&c_state), curand_uniform(&c_state));

                    auto albedo =  a_ * b_;
                    sphere_material = new lambertian(albedo);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto a_ = vec3(curand_uniform(&c_state), curand_uniform(&c_state), curand_uniform(&c_state));
                    color albedo = a_ / 2.0f + vec3(0.5, 0.5, 0.5);
                    auto fuzz = curand_uniform(&c_state) / 2.0f;
                    sphere_material = new metal(albedo, fuzz);
                } else {
                    // glass
                    sphere_material = new dielectric(1.5);                    
                }
                d_list[idx++] = new sphere(center, 0.2, sphere_material);
            }
        }
    }
    // // printf("%d\n", idx);

    *d_world = new hitable_list(d_list, idx);

    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    auto aperture = 0.1;
    auto aspect_ratio = 16.0 / 9.0;
    *d_cam = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
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

  const unsigned tx = 16, ty = 16, num_pixels = nx * ny;
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
  checkCudaErrors(cudaMalloc((void**)&d_list, (487) * sizeof(hitable*)));
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
