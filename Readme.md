Caffe-compact
==================
Caffe-compact aims to provide a self-contained CNN model testing library.

This project remove most unnecessary dependency for CNN net testing and
feature extraction. Note that we completely remove CUDA dependency in
caffe-compact.

Current dependency:
* c++11 compiler (for shared_ptr)
* google protobuf
* cblas (e.g libatlas3gf-base)

These dependencies can be satisfied on most platform including Windows and
mobile. It makes Caffe-compact much easier to deploy.

This work also avoids potential license problems along with the 
third-party libraris when release your caffe CNN model.

Difference
==================
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
* Use Eigen to avoid cblas library deployment
* integrate protobuf


Yuheng Chen, 2014
chyh1990@gmail.com
