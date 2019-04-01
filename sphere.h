#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
public:
	__device__ sphere() {}
	__device__ sphere(vec3 c, float r/*, material* m*/) : center(c), radius(r)/*, mat_ptr(m)*/ {};

	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

	vec3 center;
	float radius;
	//material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 ac = r.A - center;
	float a = dot(r.B, r.B);
	float b = dot(r.B, ac);
	float c = dot(ac, ac) - radius * radius;
	float d = b * b - a * c;

	if (d > 0) {
		float temp = (-b - sqrt(d)) / a;
		if (t_min < temp && temp < t_max) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			//rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(d)) / a;
		if (t_min < temp && temp < t_max) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			//rec.mat_ptr = mat_ptr;
			return true;
		}
	}

	return false;
}

#endif
