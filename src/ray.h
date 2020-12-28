#ifndef RAYH
#define RAYH

#include "vec3.h"

class ray {
 public:
  __device__ __host__ ray() {}
  __device__ __host__ ray(const vec3& a, const vec3& b) {
    A = a;
    B = b;
  }
  __device__ __host__ vec3 origin() const {
    return A;
  }
  __device__ __host__ vec3 direction() const{
    return B;
  }
  __device__ __host__ vec3 at(float t) const {
    return A + t * B;
  }

  vec3 A, B;
};

#endif