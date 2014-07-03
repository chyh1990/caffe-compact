// Copyright 2013 Yangqing Jia

#ifndef CAFFE_LAYER_FACTORY_HPP_
#define CAFFE_LAYER_FACTORY_HPP_

#include <string>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  const std::string& type = param.type();
  if (type == "accuracy") {
    return new AccuracyLayer<Dtype>(param);
  } else if (type == "verif_accuracy") {
    return new VerificationAccuracyLayer<Dtype>(param);
  } else if (type == "bnll") {
    return new BNLLLayer<Dtype>(param);
  } else if (type == "concat") {
    return new ConcatLayer<Dtype>(param);
  } else if (type == "conv") {
    return new ConvolutionLayer<Dtype>(param);
#if 0
  } else if (type == "data") {
    return new DataLayer<Dtype>(param);
  } else if (type == "shuffle_data") {
    return new ShuffleDataLayer<Dtype>(param);
#endif
  } else if (type == "dropout") {
    return new DropoutLayer<Dtype>(param);
  } else if (type == "dropout_group") {
    return new DropoutGroupLayer<Dtype>(param);
  } else if (type == "euclidean_loss") {
    return new EuclideanLossLayer<Dtype>(param);
  } else if (type == "flatten") {
    return new FlattenLayer<Dtype>(param);
#if 0
  } else if (type == "hdf5_data") {
    return new HDF5DataLayer<Dtype>(param);
  } else if (type == "images") {
    return new ImagesLayer<Dtype>(param);
#endif
  } else if (type == "raw_image") {
    return new RawImageLayer<Dtype>(param);
  } else if (type == "im2col") {
    return new Im2colLayer<Dtype>(param);
  } else if (type == "infogain_loss") {
    return new InfogainLossLayer<Dtype>(param);
  } else if (type == "innerproduct") {
    return new InnerProductLayer<Dtype>(param);
  } else if (type == "lrn") {
    return new LRNLayer<Dtype>(param);
  } else if (type == "multinomial_logistic_loss") {
    return new MultinomialLogisticLossLayer<Dtype>(param);
  } else if (type == "padding") {
    return new PaddingLayer<Dtype>(param);
  } else if (type == "pool") {
    return new PoolingLayer<Dtype>(param);
  } else if (type == "relu") {
    return new ReLULayer<Dtype>(param);
  } else if (type == "sigmoid") {
    return new SigmoidLayer<Dtype>(param);
  } else if (type == "softmax") {
    return new SoftmaxLayer<Dtype>(param);
  } else if (type == "softmax_loss") {
    return new SoftmaxWithLossLayer<Dtype>(param);
  } else if (type == "split") {
    return new SplitLayer<Dtype>(param);
  } else if (type == "tanh") {
    return new TanHLayer<Dtype>(param);
#if 0
  } else if (type == "window_data") {
    return new WindowDataLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Unknown layer name: " << type;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}

template Layer<float>* GetLayer(const LayerParameter& param);
template Layer<double>* GetLayer(const LayerParameter& param);

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_HPP_
