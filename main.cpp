#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <thread>
#include "camera.h"
#include "hitable.h"
#include "hitablelist.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "src/thread_pool.h"
// #include "vec3.h"
#include "src/vec3e.h"

using namespace std::chrono;

hitable* random_scene() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);
  int n = 500;
  hitable** list = new hitable*[n + 1];
  list[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
  int i = 1;
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = dis(gen);
      vec3 center(a + 0.9 * dis(gen), 0.2, b + 0.9 * dis(gen));
      if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
        if (choose_mat < 0.8) { // diffuse
          list[i++] = new sphere(
              center, 0.2, new lambertian(vec3(dis(gen) * dis(gen), dis(gen) * dis(gen), dis(gen) * dis(gen))));
        } else if (choose_mat < 0.95) { // metal
          list[i++] = new sphere(
              center,
              0.2,
              new metal(vec3(0.5 * (1 + dis(gen)), 0.5 * (1 + dis(gen)), 0.5 * (1 + dis(gen))), 0.5 * dis(gen)));
        } else { // glass
          list[i++] = new sphere(center, 0.2, new dielectric(1.5));
        }
      }
    }
  }

  list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
  list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
  list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

  return new hitable_list(list, i);
}

vec3 color(const ray& r, const hitable* const world, int depth) {
  hit_record rec;

  if (world->hit(r, 0.0001, MAXFLOAT, rec)) {
    ray scattered;
    vec3 attenuation;

    if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return attenuation * color(scattered, world, depth + 1);
    } else {
      return vec3(0, 0, 0);
    }
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * vec3(1., 1., 1.) + t * vec3(.5, .7, 1.);
  }
}

void async_std(
    const int& nx,
    const int& ny,
    const int& ns,
    const hitable* const& world,
    const camera& cam,
    std::mt19937& gen,
    std::uniform_real_distribution<float>& dis,
    unsigned* result) {
  std::vector<std::future<void>> promises;
  for (int j = ny - 1; j >= 0; --j) {
    for (int i = 0; i < nx; ++i) {
      auto promise = std::async([&, j, i]() {
        vec3 col(0, 0, 0);
        for (int s = 0; s < ns; ++s) {
          float u = float(i + dis(gen)) / float(nx);
          float v = float(j + dis(gen)) / float(ny);
          ray r = cam.get_ray(u, v);
          vec3 p = r.point_at_parameter(2.0);
          col += color(r, world, 0);
        }

        col /= float(ns);

        int ir = int(255.99 * sqrt(col[0]));
        int ig = int(255.99 * sqrt(col[1]));
        int ib = int(255.99 * sqrt(col[2]));

        result[j * nx * 3 + i * 3] = ir;
        result[j * nx * 3 + i * 3 + 1] = ig;
        result[j * nx * 3 + i * 3 + 2] = ib;
      });

      promises.emplace_back(std::move(promise));
    }
  }
  for (auto& p : promises)
    p.wait();
}

void async_std_pool_V1(
    const int& nx,
    const int& ny,
    const int& ns,
    const hitable* const& world,
    const camera& cam,
    std::mt19937& gen,
    std::uniform_real_distribution<float>& dis,
    unsigned* result) {
  thread_pool t_pool;
  for (int j = ny - 1; j >= 0; --j) {
    for (int i = 0; i < nx; ++i) {
      t_pool._async([&, j, i]() {
        vec3 col(0, 0, 0);
        for (int s = 0; s < ns; ++s) {
          float u = float(i + dis(gen)) / float(nx);
          float v = float(j + dis(gen)) / float(ny);
          ray r = cam.get_ray(u, v);
          vec3 p = r.point_at_parameter(2.0);
          col += color(r, world, 0);
        }

        col /= float(ns);

        int ir = int(255.99 * sqrt(col[0]));
        int ig = int(255.99 * sqrt(col[1]));
        int ib = int(255.99 * sqrt(col[2]));

        result[j * nx * 3 + i * 3] = ir;
        result[j * nx * 3 + i * 3 + 1] = ig;
        result[j * nx * 3 + i * 3 + 2] = ib;
      });

      // promises.emplace_back(std::move(promise));
    }
  }
}

int main() {
  auto start = high_resolution_clock::now();
  const int nx = 800;
  const int ny = 400;
  const int ns = 100;

  const vec3 lower_left_corner(-2., -1., -1.);
  const vec3 horizontal(4., 0., 0.);
  const vec3 vertical(0., 2., 0.);
  const vec3 origin(0., 0., 0.);

  const hitable* const world = random_scene();

  const vec3 lookfrom(13, 2, 3);
  const vec3 lookat(0, 0, 0);
  const float dist_to_focus = 10.0;
  const float aperture = 0.1;
  const camera cam(lookfrom, lookat, vec3(0, 1, 0), 20, float(nx) / float(ny), aperture, dist_to_focus);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1);

  auto* result = new unsigned[ny * nx * 3];

  // async_std(nx, ny, ns, world, cam, gen, dis, result);
  async_std_pool_V1(nx, ny, ns, world, cam, gen, dis, result);

  std::ofstream myfile;
  myfile.open("test.ppm");

  myfile << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; --j) {
    for (int i = 0; i < nx; ++i) {
      myfile << result[j * nx * 3 + i * 3] << " " << result[j * nx * 3 + i * 3 + 1] << " "
             << result[j * nx * 3 + i * 3 + 2] << "\n";
    }
  }
  myfile.close();
  auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start);
  std::cout << "Ended in " << duration_cast<seconds>(duration).count() << " s" << std::endl;
  // delete world;
  // delete list[0];
  // delete list[1];
  delete[] result;
  return 0;
}