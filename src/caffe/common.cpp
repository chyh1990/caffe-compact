// Copyright 2013 Yangqing Jia

#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;


int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
#ifdef _MSC_VER
  pid = 0x32423;
#else
  pid = getpid();
#endif
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


Caffe::Caffe()
    : mode_(Caffe::CPU), phase_(Caffe::TRAIN){
}

Caffe::~Caffe() {
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  // Yangqing's note: simply setting the generator seed does not seem to
  // work on the tesla K20s, so I wrote the ugly reset thing below.
}

void Caffe::SetDevice(const int device_id) {
	LOG(INFO) << "Caffe-compact only support CPU";
}

void Caffe::DeviceQuery() {
	LOG(INFO) << "Caffe-compact only support CPU";
    return;
}

}  // namespace caffe
