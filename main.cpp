#include <fstream>
#include <iostream>
#include <random>
#include "camera.h"
#include "hitable.h"
#include "hitablelist.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

vec3 color(const ray& r, hitable* world) {
  hit_record rec;

  if (world->hit(r, 0.0, MAXFLOAT, rec)) {
    return 0.5 * vec3(rec.normal.x() + 1, rec.normal.y() + 1, rec.normal.z() + 1);
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t) * vec3(1., 1., 1.) + t * vec3(.5, .7, 1.);
  }
}

int main() {
  int nx = 200;
  int ny = 100;
  int ns = 100;

  std::ofstream myfile;
  myfile.open("test.ppm");

  myfile << "P3\n" << nx << " " << ny << "\n255\n";

  vec3 lower_left_corner(-2., -1., -1.);
  vec3 horizontal(4., 0., 0.);
  vec3 vertical(0., 2., 0.);
  vec3 origin(0., 0., 0.);

  hitable* list[2];
  list[0] = new sphere(vec3(0, 0, -1), 0.5);
  list[1] = new sphere(vec3(0, -100.5, -1), 100);
  hitable* world = new hitable_list(list, 2);
  camera cam;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  for (int j = ny - 1; j >= 0; --j) {
    for (int i = 0; i < nx; ++i) {
      vec3 col(0, 0, 0);
      for (int s = 0; s < ns; ++s) {
        float u = float(i + dis(gen)) / float(nx);
        float v = float(j + dis(gen)) / float(ny);
        ray r = cam.get_ray(u, v);
        vec3 p = r.point_at_parameter(2.0);
        col += color(r, world);
      }

      col /= float(ns);

      int ir = int(255.99 * col[0]);
      int ig = int(255.99 * col[1]);
      int ib = int(255.99 * col[2]);

      myfile << ir << " " << ig << " " << ib << "\n";
    }
  }
  myfile.close();
  delete world;
  delete list[0];
  delete list[1];
  return 0;
}