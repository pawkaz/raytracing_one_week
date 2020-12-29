#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
 public:
  __host__ __device__ vec3() {}
  __host__ __device__ vec3(float e0, float e1, float e2) {
    e[0] = e0;
    e[1] = e1;
    e[2] = e2;
  }
  __host__ __device__ inline float x() const {
    return e[0];
  }
  __host__ __device__ inline float y() const {
    return e[1];
  }
  __host__ __device__ inline float z() const {
    return e[2];
  }
  __host__ __device__ inline float r() const {
    return e[0];
  }
  __host__ __device__ inline float g() const {
    return e[1];
  }
  __host__ __device__ inline float b() const {
    return e[2];
  }

  __host__ __device__ inline const vec3& operator+() const {
    return *this;
  }
  __host__ __device__ inline vec3 operator-() const {
    return vec3(-e[0], -e[1], -e[2]);
  }
  __host__ __device__ inline float operator[](int i) const {
    return e[i];
  }
  __host__ __device__ inline float& operator[](int i) {
    return e[i];
  };

  __host__ __device__ inline vec3& operator+=(const vec3& v2);
  __host__ __device__ inline vec3& operator-=(const vec3& v2);
  __host__ __device__ inline vec3& operator*=(const vec3& v2);
  __host__ __device__ inline vec3& operator/=(const vec3& v2);
  __host__ __device__ inline vec3& operator*=(const float t);
  __host__ __device__ inline vec3& operator/=(const float t);

  __host__ __device__ inline float length() const {
    return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  }
  __host__ __device__ inline float length_squared() const {
    return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
  }
  __host__ __device__ inline void make_unit_vector();

  __host__ __device__ bool near_zero() const {
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
  }

  float e[3];
};

inline std::istream& operator>>(std::istream& is, vec3& t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
  float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
  return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
  return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
  return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
  return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
  return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
  return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {
  return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
  return vec3(
      (v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
      (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
      (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v) {
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v) {
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v) {
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
  float k = 1.0 / t;

  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
  return v / v.length();
}

__device__ vec3 random_in_unit_sphere(curandState* r_state, bool unit_length = true) {
  vec3 p;
  do {
    p = 2.0f * vec3(curand_uniform(r_state), curand_uniform(r_state), curand_uniform(r_state)) - vec3(1, 1, 1);
  } while (p.length_squared() >= 1.0f);
  return unit_length ? unit_vector(p) : p;
}

__device__ inline float degrees_to_radians(double degrees) {
    return degrees * 3.141592654f / 180.0;
}

using point3 = vec3;
using color = vec3;

#endif