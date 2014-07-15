// Copyright 2013 Yangqing Jia

#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

//#include "pthread.h"
//#include "boost/scoped_ptr.hpp"

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


// The neuron layer is a specific type of layers that just works on single
// celements.
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
};


template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype> {
 public:
  explicit ReLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class TanHLayer : public NeuronLayer<Dtype> {
 public:
  explicit TanHLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype> {
 public:
  explicit SigmoidLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class BNLLLayer : public NeuronLayer<Dtype> {
 public:
  explicit BNLLLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};


template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  shared_ptr<SyncedMemory> rand_vec_;
  float threshold_;
  float scale_;
  unsigned int uint_thres_;
};

template <typename Dtype>
class DropoutGroupLayer : public NeuronLayer<Dtype> {
 public:
  explicit DropoutGroupLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  void UpdateMask();
  void ShareMask(DropoutGroupLayer *dropout) {
	  CHECK_EQ(dropout->rand_vec_->size(), rand_vec_->size());
	  rand_vec_ = dropout->rand_vec_;
  }
  void UpscaleMaskFrom(DropoutGroupLayer *dropout);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  shared_ptr<SyncedMemory> rand_vec_;
  float threshold_;
  shared_ptr<SyncedMemory> scale_;
  unsigned int uint_thres_;

  int NUM_;
  int HEIGHT_;
  int WIDTH_;
};



template <typename Dtype>
class SplitLayer : public Layer<Dtype> {
 public:
  explicit SplitLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int count_;
};


template <typename Dtype>
class FlattenLayer : public Layer<Dtype> {
 public:
  explicit FlattenLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int count_;
};


template <typename Dtype>
class InnerProductLayer : public Layer<Dtype> {
 public:
  explicit InnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int M_;
  int K_;
  int N_;
  bool biasterm_;
  shared_ptr<SyncedMemory> bias_multiplier_;
};


template <typename Dtype>
class PaddingLayer : public Layer<Dtype> {
 public:
  explicit PaddingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  unsigned int PAD_;
  int NUM_;
  int CHANNEL_;
  int HEIGHT_IN_;
  int WIDTH_IN_;
  int HEIGHT_OUT_;
  int WIDTH_OUT_;
};


template <typename Dtype>
class LRNLayer : public Layer<Dtype> {
 public:
  explicit LRNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  // scale_ stores the intermediate summing results
  Blob<Dtype> scale_;
  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  int num_;
  int channels_;
  int height_;
  int width_;
};


template <typename Dtype>
class Im2colLayer : public Layer<Dtype> {
 public:
  explicit Im2colLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int KSIZE_;
  int STRIDE_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int PAD_;
};

template <typename Dtype>
class PoolingLayer : public Layer<Dtype> {
 public:
  explicit PoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  int KSIZE_;
  int STRIDE_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int POOLED_HEIGHT_;
  int POOLED_WIDTH_;
  Blob<float> rand_idx_;
};


template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
 public:
  explicit ConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> col_bob_;

  int KSIZE_;
  int STRIDE_;
  int NUM_;
  int CHANNELS_;
  int PAD_;
  int HEIGHT_;
  int WIDTH_;
  int NUM_OUTPUT_;
  int GROUP_;
  Blob<Dtype> col_buffer_;
  Blob<Dtype> out_buffer_;
  shared_ptr<SyncedMemory> bias_multiplier_;
  bool biasterm_;
  int M_;
  int K_;
  int N_;
  int NTILE_WIDTH_;
  int NTILE_HEIGHT_;
  int TILE_WIDTH_;
  int TILE_HEIGHT_;
};

template <typename Dtype>
class ConcatLayer : public Layer<Dtype> {
 public:
  explicit ConcatLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> col_bob_;

  int COUNT_;
  int NUM_;
  int CHANNELS_;
  int HEIGHT_;
  int WIDTH_;
  int concat_dim_;
};

#if 0
// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* DataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class DataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* DataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
};

template <typename Dtype>
class ShuffleDataLayer : public Layer<Dtype> {
 public:
  explicit ShuffleDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ShuffleDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  void CopyDataPtrFrom(const ShuffleDataLayer<Dtype>& source){
	  prefetch_data_ = source.prefetch_data_;
	  prefetch_label_ = source.prefetch_label_;
	  idx_[0] = source.idx_[0];
	  idx_[1] = source.idx_[1];
  }

  /*
  int GetCurrent() const {return current_;}
  void SetCurrent(int cur) {
	  CHECK_LT(cur, idx_[0]->size());
	  CHECK_GE(cur, 0);
	  current_ = cur;
  }
  */
  void SetOutputChannel(int n){
	  CHECK(n == 0 || n == 1);
	  OUTPUT_CHANNEL_ = n;
	  LOG(INFO) << "Set output ch " << n;
  }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<leveldb::DB> db_;
  shared_ptr<leveldb::Iterator> iter_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  size_t datum_size_;

  size_t DATA_COUNT_;
  int OUTPUT_CHANNEL_;
  //Blob fail to handle large data
  //shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<vector<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  int current_[2];
  shared_ptr<vector<int> > idx_[2];

};
#endif

#if 0
// This function is used to create a pthread that prefetches the data.
template <typename Dtype>
void* ImagesLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class ImagesLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* ImagesLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit ImagesLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~ImagesLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;
  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
};
#endif

template <typename Dtype>
class RawImageLayer: public Layer<Dtype> {

 public:
  explicit RawImageLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~RawImageLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  int datum_channels_;
  int datum_height_;
  int datum_width_;
  int datum_size_;
};


#if 0
template <typename Dtype>
class HDF5DataLayer : public Layer<Dtype> {
 public:
  explicit HDF5DataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~HDF5DataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual void load_hdf5_file_data(const char* filename);

  std::vector<std::string> hdf_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  hsize_t current_row_;

  Blob<Dtype> data_blob_;
  Blob<Dtype> label_blob_;
};
#endif

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  // sum_multiplier is just used to carry out sum using blas
  Blob<Dtype> sum_multiplier_;
  // scale is an intermediate blob to hold temporary results.
  Blob<Dtype> scale_;
};


template <typename Dtype>
class MultinomialLogisticLossLayer : public Layer<Dtype> {
 public:
  explicit MultinomialLogisticLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
};

template <typename Dtype>
class InfogainLossLayer : public Layer<Dtype> {
 public:
  explicit InfogainLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), infogain_() {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> infogain_;
};


// SoftmaxWithLossLayer is a layer that implements softmax and then computes
// the loss - it is preferred over softmax + multinomiallogisticloss in the
// sense that during training, this will produce more numerically stable
// gradients. During testing this layer could be replaced by a softmax layer
// to generate probability outputs.
template <typename Dtype>
class SoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), softmax_layer_(new SoftmaxLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying softmax layer forward and backward.
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
};

template <typename Dtype>
class VerificationLossLayer : public Layer<Dtype> {
 public:
  explicit VerificationLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  void SetThreshold(Dtype t) { M_ = t; }
  Dtype GetThreshold() { return M_ ; }

  void GetMeanDistance(vector<Dtype> &dists){ 
	  dists.clear();
	  Dtype avg[2] = {0.};
	  int cnt[2] = {0};
	  CHECK_EQ(distance_.size(), same_.size());
	  for(size_t i=0;i<distance_.size();i++){
		  int s = same_[i];
		  avg[s] += distance_[i];
		  cnt[s] ++;
	  }

	  for(int i=0;i<2;i++)
	 	 dists.push_back(avg[i] / std::max(cnt[i], 1));
  }
  void ResetDistanceStat(){
	  distance_.clear();
	  same_.clear();
  }
  Dtype CalcThreshold(bool update);
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  Blob<Dtype> diffy1_;
  Blob<Dtype> diffy2_;

  Dtype M_;
  Dtype LAMDA_;

  std::vector<Dtype> distance_;
  std::vector<int>   same_;
};


template <typename Dtype>
class EuclideanLossLayer : public Layer<Dtype> {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), difference_() {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  // The loss layer will do nothing during forward - all computation are
  // carried out in the backward pass.
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return; }
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  Blob<Dtype> difference_;
};


template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> {
 public:
  explicit AccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // The accuracy layer should not be used to compute backward operations.
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
    return Dtype(0.);
  }
};

template <typename Dtype>
class VerificationAccuracyLayer : public Layer<Dtype> {
 public:
  explicit VerificationAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

  void SetThreshold(Dtype t) { M_ = t; }
  Dtype GetThreshold(Dtype t) { return M_ ; }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  // The accuracy layer should not be used to compute backward operations.
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
    NOT_IMPLEMENTED;
    return Dtype(0.);
  }
  Blob<Dtype> diffy_;

  Dtype M_;
};


#if 0
// This function is used to create a pthread that prefetches the window data.
template <typename Dtype>
void* WindowDataLayerPrefetch(void* layer_pointer);

template <typename Dtype>
class WindowDataLayer : public Layer<Dtype> {
  // The function used to perform prefetching.
  friend void* WindowDataLayerPrefetch<Dtype>(void* layer_pointer);

 public:
  explicit WindowDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~WindowDataLayer();
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  pthread_t thread_;
  shared_ptr<Blob<Dtype> > prefetch_data_;
  shared_ptr<Blob<Dtype> > prefetch_label_;
  Blob<Dtype> data_mean_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM };
  vector<vector<float> > fg_windows_;
  vector<vector<float> > bg_windows_;
};
#endif

}

#endif  // CAFFE_VISION_LAYERS_HPP_
