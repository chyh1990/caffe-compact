// Copyright 2014 Aravindh Mahendran
// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#ifdef __SSE2__
#include "caffe/fmath.hpp"
#define EXP(x) fmath::exp(x)
#else
#define cast_uint32_t static_cast<uint32_t>
static inline float
fastpow2 (float p)
{
	float offset = (p < 0) ? 1.0f : 0.0f;
	float clipp = (p < -126) ? -126.0f : p;
	int w = clipp;
	float z = clipp - w + offset;
	union { uint32_t i; float f; } v = { cast_uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

	return v.f;
}

	static inline float
fastexp (float p)
{
	return fastpow2 (1.442695040f * p);
}
#define EXP(x) fastexp(x)
#endif

namespace caffe {

template <typename Dtype>
	void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			vector<Blob<Dtype>*>* top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = (*top)[0]->mutable_cpu_data();
		Dtype exp2x;
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			exp2x = EXP(2*bottom_data[i]);
			top_data[i] = (exp2x - Dtype(1))/(exp2x + Dtype(1));
		}
	}

template <typename Dtype>
	Dtype TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const bool propagate_down,
			vector<Blob<Dtype>*>* bottom) {
		if (propagate_down) {
			const Dtype* bottom_data = (*bottom)[0]->cpu_data();
			const Dtype* top_diff = top[0]->cpu_diff();
			Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
			const int count = (*bottom)[0]->count();
			Dtype exp2x;
			Dtype tanhx;
			for (int i = 0; i < count; ++i) {
				exp2x = exp(2*bottom_data[i]);
				tanhx = (exp2x - Dtype(1))/(exp2x + Dtype(1));
				bottom_diff[i] = top_diff[i] * (1 - tanhx*tanhx);
			}
		}
		return Dtype(0);
	}

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
