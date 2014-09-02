// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>

#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CROP_WINSIZE 39
#define CROP_PADDING 2.5

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;

float getMean( float * p )
{
	float ans;
	for(int i = 0 ; i < CROP_WINSIZE * CROP_WINSIZE ; i++, p++)
		ans += *p;
	ans = ans/ float( CROP_WINSIZE * CROP_WINSIZE );
	return ans;
}

float getStd( float * p , float mean)
{
	float ans;
	for(int i = 0 ; i < CROP_WINSIZE * CROP_WINSIZE ; i++, p++)
		ans += ( (*p-mean) * (*p-mean) );
	ans = ans/ float( CROP_WINSIZE * CROP_WINSIZE - 1);
	ans = sqrt( ans );
	return ans;
}
void getZscore( Mat & img, int left, int right, int top, int bottom, float * & score )
{
	if( img.type()==CV_8UC3 )
	{
		std::cerr << "warning! a color image input" << std::endl;
		cv::cvtColor( img , img , CV_RGB2GRAY );
	}

	double scale = (right - left) / double( CROP_WINSIZE );
	
	left  -= int( scale * CROP_PADDING );
	right += int( scale * CROP_PADDING );
	top   -= int( scale * CROP_PADDING );
	bottom+= int( scale * CROP_PADDING );
		
	if( top<0 || left < 0 || right >= img.cols || bottom >= img.rows )
	{
		std::cerr << "warning! invalid bounding box " << std::endl;
		return;
	}
	
	
	Mat patch = img( Range( top, bottom	), Range( left, right ) );
	cv::resize( patch , patch, Size( CROP_WINSIZE, CROP_WINSIZE ) );
	
	patch.convertTo( patch, CV_32F );	
	
	float mu = getMean(  patch.ptr<float>() );
	float sigma = getStd(  patch.ptr<float>() , mu);
	
	score = new( float[ CROP_WINSIZE * CROP_WINSIZE ] );
	
	float * p_patch = patch.ptr<float>();
	
	for(int i = 0 ; i < CROP_WINSIZE * CROP_WINSIZE ; i++)
		score[i] = ( p_patch[i] - mu ) / sigma;
}


template <typename Dtype>
static void save_blob(const string& fn, Blob<Dtype> *b){
	LOG(INFO) << "Saving " << fn;
	FILE *f = fopen(fn.c_str(), "wb");
	CHECK(f != NULL);
	fwrite(b->cpu_data(), sizeof(Dtype), b->count(), f);
	fclose(f);
}

static void draw(const float *buf, const float *pt){
	const int ph = 39, pw = 39;
	const float scale = 4.0f;
	cv::Mat m = cv::Mat::zeros(ph, pw, CV_32FC1);
	memcpy(m.data, buf, sizeof(float)*pw*ph);
	cv::Mat dsp;
	cv::normalize(m, dsp, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::resize(dsp, dsp, cv::Size(), scale, scale);
	cv::cvtColor(dsp, dsp, CV_GRAY2BGR);

#if 1
	for(int i=0;i<5;i++){
		const float *t = pt + 2*i;
		cv::circle(dsp, cv::Point(t[0]*scale, t[1]*scale), 2, cv::Scalar(255,0,0), 2);
	}
#endif
		cv::imshow("A", dsp);
		cv::waitKey(0);
}


int main(int argc, char** argv) {
	if (argc < 3) {
		LOG(ERROR) << "test_net net_proto pretrained_net_proto iterations inputbin output_dir"
			<< " [CPU/GPU]";
		return 0;
	}

	LogMessage::Enable(true);
	Caffe::set_phase(Caffe::TEST);
	Caffe::set_mode(Caffe::CPU);

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

#if 0
	SolverState state;
	std::string state_file = std::string(argv[2]) + ".solverstate";
	ReadProtoFromBinaryFile(state_file, &state);
#endif

	vector<Blob<float>*> dummy_blob_input_vec;

	//save layer
	int feature_layer_idx = -1;
	int data_layer_idx = -1;
	for(int i=0;i<caffe_test_net.layer_names().size();i++)
		if(caffe_test_net.layer_names()[i] == "ip2"){
			feature_layer_idx = i;
			break;
		}
	for(int i=0;i<caffe_test_net.layer_names().size();i++)
		if(caffe_test_net.layer_names()[i] == "image_input"){
			data_layer_idx = i;
			break;
		}

	CHECK_NE(feature_layer_idx, -1);
	CHECK_NE(data_layer_idx, -1);
	LOG(INFO) << "Data layer: " << data_layer_idx;
	LOG(INFO) << "Feature layer: " << feature_layer_idx;

	Blob<float>* output = caffe_test_net.top_vecs()[feature_layer_idx][0],
		*data_blob = caffe_test_net.top_vecs()[data_layer_idx][0];
	RawImageLayer<float> *data_layer = dynamic_cast<RawImageLayer<float>* >(caffe_test_net.layers()[data_layer_idx].get());
	CHECK(data_layer != 0);

	LOG(INFO) << "OUTPUT BLOB dim: " << output->num() << ' '
		<< output->channels() << ' '
		<< output->width() << ' '
		<< output->height();
	const int ih = data_blob->height(), iw = data_blob->width(), ic = data_blob->channels();
	//double buf[ih*iw*ic];
	FILE *finput = fopen(argv[3], "r");
	CHECK(finput != NULL);
	for (;;) {
		char fn[1024];
		int l,r,t,b;
		int nread = fscanf(finput, "%s%d%d%d%d", fn, &l, &r, &t, &b);
		if(nread != 5)
			break;
		cv::Mat mat = cv::imread(fn);
		if(!mat.data){
			printf("%s\n", fn);
			continue;
		}
		cv::cvtColor(mat, mat, CV_BGR2GRAY);
		float * p = 0;
		getZscore( mat, l, r, t, b, p);
		CHECK(p != NULL);

		float *d = data_blob->mutable_cpu_data();
		size_t len = ih * iw * ic;
		for(int j = 0; j < data_blob->num(); j++){
			memcpy(d, p, sizeof(float)*CROP_WINSIZE*CROP_WINSIZE);
			/*
			   size_t nread = fread(buf, sizeof(double), len, finput);
			   CHECK_EQ(nread, len);
			   for(int k=0;k<len;k++){
			   d[k] = buf[k];
			   }
			   d += len;
			   */
		}
		const vector<Blob<float>*>& result =
			caffe_test_net.Forward(dummy_blob_input_vec);

		printf("%s %d %d %d %d ", fn, l, r, t, b);
		const float *pt = output->cpu_data();
		for(int i=0;i<output->num();i++){
			for(int j=0;j<output->channels();j++)
				printf("%f\t", pt[j]);
			printf("\n");
		}
		fflush(stdout);

		//draw(p, pt);
		delete [] p;

		//sprintf(output_dir, "%s/feat_%05d", argv[4], i);
		//save_blob(output_dir, output);

		//test_accuracy += result[0]->cpu_data()[0];
		//LOG(ERROR) << "Batch " << i << ", accuracy: " << result[0]->cpu_data()[0];
	}
	fclose(finput);
	//test_accuracy /= total_iter;
	//LOG(ERROR) << "Test accuracy:" << test_accuracy;

	return 0;
}

