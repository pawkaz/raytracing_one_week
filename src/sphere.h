#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "material.h"

class sphere : public hitable {
 public:
  __device__ sphere() {}
  __device__ sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat(m){};
  __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
  vec3 center;
  float radius;
  material* mat;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = half_b*half_b - a*c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.normal = (rec.p - center) / radius;
    rec.mat_ptr = mat;
    rec.set_face_normal(r, rec.normal);

    return true;
}

#endif
