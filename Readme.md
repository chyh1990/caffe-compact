Caffe-compact
==================

caffe_version: 9e970c17c337e770f0df396d3b7882f6f97cf308

Caffe-compact aims to provide a self-contained CNN model testing library.

This project remove most unnecessary dependency for CNN net testing and
feature extraction. Note that we completely remove CUDA dependency in
caffe-compact.

Current dependency:
* c++11 compiler (for shared_ptr)
* google protobuf

Optional dependency:
* cblas (e.g. libatlas3gf-base)
* Eigen3

You can select an matrix backend by setting the USE_EIGEN environment in the
Makefile.

These dependencies can be satisfied on most platform including Windows and
mobile. It makes Caffe-compact much easier to deploy.

This work also avoids potential license problems along with the 
third-party libraris when release your caffe CNN model.

Difference
==================
The original project can be found at: https://github.com/BVLC/caffe
Caffe-compact only support a subset of functionality of caffe:

* CNN forward pass only 
* CPU only
* Raw image input only

Performance
==================
MKL has performance problem when dealing with small matrix (e.g. testing your
model on only one input image), especially multithreading is enabled. Atlas or
other open source BLAS implementation may perform better.

TODO: benchmark

Future Work
==================
* integrate protobuf


Yuheng Chen, 2014

chyh1990@gmail.com

