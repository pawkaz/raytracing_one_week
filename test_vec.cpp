#include <iostream>
// #include "vec3.h"
// #include "src/vec3e.h"
#include <chrono>
#include <eigen3/Eigen/Dense>
int main() {
  //   vec3 o(2, 1, 40);
  Eigen::Array3f o(2, 1, 40);
  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned i = 0; i < 400000000; ++i) {
    o += o;
    o *= o;
  }

  printf(
      "Time passed %ld \n",
      std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count());

  return 0;
}