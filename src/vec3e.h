#ifndef vec3H
#define vec3H

#include <math.h>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
// #include "../vec3.h"
class vec3 {
 public:
  vec3() {}
  vec3(float e0, float e1, float e2) {
    e << e0, e1, e2;
  }
  vec3(const vec3& v) {
    e = v.e;
  }

  vec3(const Eigen::Array3f& v) {
    e = v;
  }

  vec3& operator=(Eigen::Array3f&& v) {
    e = std::move(v);
    return *this;
  }

  inline float x() const {
    return e(0);
  }
  inline float y() const {
    return e(1);
  }
  inline float z() const {
    return e(2);
  }
  inline float r() const {
    return e(0);
  }
  inline float g() const {
    return e(1);
  }
  inline float b() const {
    return e(2);
  }

  inline const vec3& operator+() const {
    return *this;
  }
  inline vec3 operator-() const {
    return vec3(-e);
  }
  inline float operator[](int i) const {
    return e(i);
  }
  inline float& operator[](int i) {
    return e(i);
  }

  inline vec3& operator+=(const vec3& v2);
  inline vec3& operator-=(const vec3& v2);
  inline vec3& operator*=(const vec3& v2);
  inline vec3& operator/=(const vec3& v2);
  inline vec3& operator*=(const float t);
  inline vec3& operator/=(const float t);

  inline float length() const {
    return sqrt(e.square().sum());
  }
  inline float squared_length() const {
    return e.square().sum();
  }
  inline void make_unit_vector();

  // float e[3];

  Eigen::Array3f e;
};

inline std::istream& operator>>(std::istream& is, vec3& t) {
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

inline void vec3::make_unit_vector() {
  float k = 1.0 / this->length();
  e *= k;
}

inline vec3 operator+(const vec3& v1, const vec3& v2) {
  return vec3(v1.e + v2.e);
}

inline vec3 operator-(const vec3& v1, const vec3& v2) {
  return vec3(v1.e - v2.e);
}

inline vec3 operator*(const vec3& v1, const vec3& v2) {
  return vec3(v1.e * v2.e);
}

inline vec3 operator/(const vec3& v1, const vec3& v2) {
  return vec3(v1.e / v2.e);
}

inline vec3 operator*(const float t, const vec3& v) {
  auto a = t * v.e;
  return vec3(t * v.e);
}

inline vec3 operator/(const vec3 v, const float t) {
  return vec3(v.e / t);
}

inline vec3 operator*(const vec3& v, float t) {
  return vec3(v.e * t);
}

inline float dot(const vec3& v1, const vec3& v2) {
  return (v1.e * v2.e).sum();
}

inline vec3 cross(const vec3& v1, const vec3& v2) {
  // return vec3(v1.e.cross(v2.e));
  return vec3(v1.e.matrix().cross(v2.e.matrix()));
  // return vec3(
  //     v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
  //     v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
  //     v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

inline vec3& vec3::operator+=(const vec3& v) {
  e += v.e;
  return *this;
}

inline vec3& vec3::operator*=(const vec3& v) {
  e *= v.e;
  return *this;
}

inline vec3& vec3::operator/=(const vec3& v) {
  e /= v.e;
  return *this;
}

inline vec3& vec3::operator-=(const vec3& v) {
  e -= v.e;
  return *this;
}

inline vec3& vec3::operator*=(const float t) {
  e *= t;
  return *this;
}

inline vec3& vec3::operator/=(const float t) {
  float k = 1.0f / t;

  e *= k;
  return *this;
}

inline vec3 unit_vector(vec3 v) {
  return v / v.length();
}

#endif