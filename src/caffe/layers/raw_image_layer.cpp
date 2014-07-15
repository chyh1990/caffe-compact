// Copyright 2013 Yangqing Jia

#include <stdint.h>

#include <string>
#include <vector>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::pair;

namespace caffe {

template <typename Dtype>
RawImageLayer<Dtype>::~RawImageLayer<Dtype>() {
  // Finally, join the thread
}

template <typename Dtype>
void RawImageLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Input Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Input Layer takes two blobs as output.";
  // datum size
  datum_height_ = this->layer_param_.new_height();
  datum_width_ = this->layer_param_.new_width();
  datum_channels_ = this->layer_param_.new_channels();
  datum_size_ = datum_channels_ * datum_height_ * datum_width_;
  // Read the file with filenames and labels
  (*top)[0]->Reshape(
        this->layer_param_.batchsize(), datum_channels_, 
	datum_height_, datum_width_);
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
 }

template <typename Dtype>
void RawImageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->cpu_data();
  (*top)[1]->cpu_data();
 }

#if 0
template <typename Dtype>
void RawImageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  (*top)[0]->gpu_data();
  (*top)[1]->gpu_data();
}
#endif

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype RawImageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

#if 0
template <typename Dtype>
Dtype RawImageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}
#endif

INSTANTIATE_CLASS(RawImageLayer);

}  // namespace caffe
