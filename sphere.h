#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
 public:
  sphere() {}
  sphere(vec3 center, float r) : center(center), radius(r) {}
  virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
  vec3 center;
  float radius;
};

bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0 * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4.0 * a * c;
  if (discriminant > 0.) {
    for (float coeff = -1; coeff < 2; coeff += 2.) {
      float temp = (-b + coeff * sqrt(discriminant)) / (2.0 * a);
      if (temp > tmin and temp < tmax) {
        rec.t = temp;
        rec.p = r.point_at_parameter(temp);
        rec.normal = (rec.p - this->center) / this->radius;
        return true;
      }
    }
  }
  return false;
}

#endif