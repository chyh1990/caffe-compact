// Copyright 2014 Yuheng Chen

#include <vector>
#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutGroupLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  const int mask_count = bottom[0]->count() / bottom[0]->channels();
  const int mask_size = bottom[0]->width() * bottom[0]->height();
  NUM_ = bottom[0]->num();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  // Set up the cache for random number generation
  rand_vec_.reset(new SyncedMemory(mask_count * sizeof(int)));
  threshold_ = this->layer_param_.dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  Dtype scale = 1. / (1. - threshold_);
  scale_.reset(new SyncedMemory(NUM_ * sizeof(Dtype)));
  uint_thres_ = (unsigned int)((mask_size * (1. - threshold_)) + 0.5);

  int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
  Dtype *scale_ptr = reinterpret_cast<Dtype*>(scale_->mutable_cpu_data());
  for(int n = 0; n < bottom[0]->num(); n++){
  	for(int i = 0; i < mask_size; i++)
	  mask[i] = i;
	mask += mask_size;
	scale_ptr[n] = scale;
  }
}

template <typename Dtype>
void DropoutGroupLayer<Dtype>::UpdateMask() {
	const int count = rand_vec_->size() / sizeof(int);
	const int mask_size = count / NUM_;
	int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
  	for(int n = 0; n < NUM_; n++){
		std::random_shuffle(mask, mask + mask_size);
		mask += mask_size;
	}
}

template <typename Dtype>
void DropoutGroupLayer<Dtype>::UpscaleMaskFrom(DropoutGroupLayer *dropout) {
	const int ksize = HEIGHT_ - dropout->HEIGHT_ + 1;
	CHECK_EQ(ksize, WIDTH_ - dropout->WIDTH_ + 1);
	CHECK(ksize > 0);
	CHECK_EQ(NUM_, dropout->NUM_);
	int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
	const int count = rand_vec_->size() / sizeof(int);
	const int mask_size = count / NUM_;
	const int* mask_downscale = reinterpret_cast<const int*>(dropout->rand_vec_->cpu_data());
	const int mask_ds_size = dropout->rand_vec_->size() / sizeof(int) / NUM_;

	//mask all
	uint_thres_ = 1;
	for(int n = 0; n < count; n++)
		mask[n] = 1;

	Dtype *scale_ptr = reinterpret_cast<Dtype*>(scale_->mutable_cpu_data());
  	for(int n = 0; n < NUM_; n++){
		int idx_ds = 0;
		for(int y = 0; y < dropout->HEIGHT_; y++){
			for(int x = 0; x < dropout->WIDTH_; x++){
				/* if kept */
				if(mask_downscale[idx_ds++] < dropout->uint_thres_){
					for(int ty = 0; ty < ksize; ty++){
						int *ptr = mask + (y + ty) * WIDTH_;	
						for(int tx = 0; tx < ksize; tx++)
							ptr[x + tx] = 0;
					} /* ty */
				}
			}
		}
		int nonzeros = 0;
		for(int y = 0; y < HEIGHT_; y++){
			int *ptr = mask + y * WIDTH_;	
			for(int x = 0; x < WIDTH_; x++){
				if(ptr[x] == 0)
					nonzeros ++;
				//fprintf(stderr, "%d ", ptr[x]);
			}
			//fprintf(stderr, "\n");
		}
			//fprintf(stderr, "\n\n");
		CHECK(nonzeros > 0);
		scale_ptr[n] = HEIGHT_ * WIDTH_ / (Dtype)nonzeros;
		mask += mask_size;
		mask_downscale +=  mask_ds_size;
	}
}

template <typename Dtype>
void DropoutGroupLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
  const int mask_size = bottom[0]->width() * bottom[0]->height();
  if (Caffe::phase() == Caffe::TRAIN) {
#if 0
	  for(int n = 0; n < bottom[0]->num(); n++){
		  for(int c = 0; c < bottom[0]->channels(); c++){
			  int i = 0;
			  for (; i < uint_thres_; ++i) {
				  int idx = mask[i];
				  top_data[idx] = bottom_data[idx] * scale_;
			  }
			  for (; i < mask_size; i++) {
				  int idx = mask[i];
				  top_data[idx] = 0.;
			  }
			  top_data += mask_size;
			  bottom_data += mask_size;
		  }
		  mask += mask_size;
	  }
#else
	  NOT_IMPLEMENTED;
#endif
  } else {
    memcpy(top_data, bottom_data, bottom[0]->count() * sizeof(Dtype));
  }
}

template <typename Dtype>
Dtype DropoutGroupLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down) {
#if 0
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int* mask = reinterpret_cast<const int*>(rand_vec_->cpu_data());
    const int mask_size = top[0]->width() * top[0]->height();
    for(int n = 0; n < top[0]->num(); n++){
	    for(int c = 0; c < top[0]->channels(); c++){
		    int i = 0;
		    for (; i < uint_thres_; ++i) {
			    int idx = mask[i];
			    bottom_diff[idx] = top_diff[idx] * scale_;
		    }
		    for (; i < mask_size; i++) {
			    int idx = mask[i];
			    bottom_diff[idx] = 0.;
		    }
		    top_diff += mask_size;
		    bottom_diff += mask_size;
	    }
	    mask += mask_size;
    }
#else
    NOT_IMPLEMENTED;
#endif
  }
  return Dtype(0);
}


INSTANTIATE_CLASS(DropoutGroupLayer);


}  // namespace caffe
