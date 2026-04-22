#include "venddb_inference.h"

#include <yolos/yolos.hpp>

#include <onnxruntime_cxx_api.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Thread-local error string
// ============================================================================

static thread_local std::string tl_last_error;

static void setError(const std::string& msg) {
	tl_last_error = msg;
}

extern "C" const char* venddb_last_error(void) {
	return tl_last_error.c_str();
}

// ============================================================================
// YAML parser (handles the project's custom YAML subset)
// ============================================================================

struct DetectorConfig {
	std::string model_path;
	float conf_threshold = 0.25f;
	float iou_threshold  = 0.45f;
	std::vector<std::string> classes;
};

static DetectorConfig parseYAML(const std::string& configPath) {
	YAML::Node doc = YAML::LoadFile(configPath);

	DetectorConfig cfg;

	if(auto n = doc["model_path"])
		cfg.model_path = n.as<std::string>();
	if(auto n = doc["conf_threshold"])
		cfg.conf_threshold = n.as<float>();
	if(auto n = doc["iou_threshold"])
		cfg.iou_threshold = n.as<float>();
	if(auto n = doc["classes"])
		for(const auto& item : n)
			cfg.classes.push_back(item.as<std::string>());

	// Resolve model path relative to the YAML directory
	if(!cfg.model_path.empty()) {
		std::filesystem::path p(cfg.model_path);
		if(p.is_relative()) {
			auto dir       = std::filesystem::path(configPath).parent_path();
			cfg.model_path = (dir / p).string();
		}
	}

	return cfg;
}

// ============================================================================
// Internal wrapper struct
// ============================================================================

struct DetectorWrapper {
	DetectorConfig config;
	std::unique_ptr<yolos::det::YOLODetector> detector;
};

// ============================================================================
// C API implementation
// ============================================================================

extern "C" VendDB_Detector venddb_load_detector(const char* config_yaml_path, int use_gpu) {
	try {
		DetectorConfig cfg = parseYAML(config_yaml_path);

		if(cfg.model_path.empty()) {
			setError("no model_path in config");
			return nullptr;
		}

		// YOLODetector reads class names from a file (one per line).
		// Write them to a temporary file, then delete after construction.
		std::string classes_tmp = std::string(config_yaml_path) + ".classes.tmp";
		{
			std::ofstream cf(classes_tmp);
			for(const auto& cls : cfg.classes)
				cf << cls << "\n";
		}

		auto wrapper      = std::make_unique<DetectorWrapper>();
		wrapper->config   = cfg;
		wrapper->detector = std::make_unique<yolos::det::YOLODetector>(cfg.model_path, classes_tmp, use_gpu != 0);

		std::remove(classes_tmp.c_str());

		return static_cast<VendDB_Detector>(wrapper.release());
	} catch(const std::exception& e) {
		setError(e.what());
		return nullptr;
	}
}

extern "C" void venddb_destroy_detector(VendDB_Detector detector) {
	delete static_cast<DetectorWrapper*>(detector);
}

// Shared post-processing: convert yolos detections → C array
static VendDB_Detection* buildResult(
	const DetectorWrapper* w, const std::vector<yolos::det::Detection>& dets, int* out_count) {
	*out_count = static_cast<int>(dets.size());
	if(dets.empty())
		return nullptr;

	auto* arr = new VendDB_Detection[dets.size()];
	for(size_t i = 0; i < dets.size(); ++i) {
		const auto& d   = dets[i];
		const auto& cfg = w->config;

		std::string class_name = "unknown";
		if(d.classId >= 0 && d.classId < static_cast<int>(cfg.classes.size()))
			class_name = cfg.classes[d.classId];

		std::strncpy(arr[i].class_name, class_name.c_str(), sizeof(arr[i].class_name) - 1);
		arr[i].class_name[sizeof(arr[i].class_name) - 1] = '\0';
		arr[i].confidence                                = d.conf;
		arr[i].x                                         = d.box.x;
		arr[i].y                                         = d.box.y;
		arr[i].width                                     = d.box.width;
		arr[i].height                                    = d.box.height;
	}
	return arr;
}

extern "C" VendDB_Detection* venddb_detect_file(
	VendDB_Detector detector, const char* image_path, float conf_threshold, float iou_threshold, int* out_count) {
	*out_count = 0;
	auto* w    = static_cast<DetectorWrapper*>(detector);
	try {
		cv::Mat frame = cv::imread(image_path);
		if(frame.empty()) {
			setError(std::string("cannot read image: ") + image_path);
			return nullptr;
		}

		float conf = conf_threshold < 0 ? w->config.conf_threshold : conf_threshold;
		float iou  = iou_threshold < 0 ? w->config.iou_threshold : iou_threshold;

		auto dets = w->detector->detect(frame, conf, iou);
		return buildResult(w, dets, out_count);
	} catch(const std::exception& e) {
		setError(e.what());
		return nullptr;
	}
}

extern "C" VendDB_Detection* venddb_detect_bytes(VendDB_Detector detector, const unsigned char* data, int data_len,
	float conf_threshold, float iou_threshold, int* out_count) {
	*out_count = 0;
	auto* w    = static_cast<DetectorWrapper*>(detector);
	try {
		std::vector<unsigned char> buf(data, data + data_len);
		cv::Mat frame = cv::imdecode(buf, cv::IMREAD_COLOR);
		if(frame.empty()) {
			setError("cannot decode image bytes");
			return nullptr;
		}

		float conf = conf_threshold < 0 ? w->config.conf_threshold : conf_threshold;
		float iou  = iou_threshold < 0 ? w->config.iou_threshold : iou_threshold;

		auto dets = w->detector->detect(frame, conf, iou);
		return buildResult(w, dets, out_count);
	} catch(const std::exception& e) {
		setError(e.what());
		return nullptr;
	}
}

extern "C" void venddb_free_detections(VendDB_Detection* detections) {
	delete[] detections;
}

// ============================================================================
// ConvNeXtV2 classifier
// ============================================================================

struct ClassifierConfig {
	std::string model_path;
	std::string labels_path;
	int input_size = 288;
};

static ClassifierConfig parseClassifierYAML(const std::string& configPath) {
	YAML::Node doc = YAML::LoadFile(configPath);
	ClassifierConfig cfg;
	if(auto n = doc["model_path"])
		cfg.model_path = n.as<std::string>();
	if(auto n = doc["labels_path"])
		cfg.labels_path = n.as<std::string>();
	if(auto n = doc["input_size"])
		cfg.input_size = n.as<int>();

	auto dir     = std::filesystem::path(configPath).parent_path();
	auto resolve = [&](std::string& p) {
		if(!p.empty() && std::filesystem::path(p).is_relative())
			p = (dir / p).string();
	};
	resolve(cfg.model_path);
	resolve(cfg.labels_path);
	return cfg;
}

// labels.json is {"label": idx, ...}; yaml-cpp parses JSON as a YAML subset.
static std::vector<std::string> loadLabels(const std::string& path) {
	YAML::Node doc = YAML::LoadFile(path);
	std::vector<std::string> labels(doc.size());
	for(auto it = doc.begin(); it != doc.end(); ++it)
		labels[it->second.as<int>()] = it->first.as<std::string>();
	return labels;
}

static std::vector<float> preprocessImage(const cv::Mat& bgr, int input_size) {
	cv::Mat rgb;
	cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

	// Resize so the shorter side equals input_size
	int h = rgb.rows, w = rgb.cols;
	cv::Size newSize = (h < w) ? cv::Size((int)std::lround((float)w * input_size / h), input_size)
							   : cv::Size(input_size, (int)std::lround((float)h * input_size / w));
	cv::resize(rgb, rgb, newSize, 0, 0, cv::INTER_LINEAR);

	// Center crop
	int y0 = (rgb.rows - input_size) / 2;
	int x0 = (rgb.cols - input_size) / 2;
	rgb    = rgb(cv::Rect(x0, y0, input_size, input_size)).clone();

	// float32 / 255, ImageNet mean/std normalisation
	cv::Mat flt;
	rgb.convertTo(flt, CV_32FC3, 1.0f / 255.0f);

	static const float mean[3] = { 0.485f, 0.456f, 0.406f };
	static const float std_[3] = { 0.229f, 0.224f, 0.225f };
	std::vector<cv::Mat> ch(3);
	cv::split(flt, ch);
	for(int c = 0; c < 3; c++)
		ch[c] = (ch[c] - mean[c]) / std_[c];

	// HWC -> CHW flat vector
	std::vector<float> data(3 * input_size * input_size);
	for(int c = 0; c < 3; c++)
		for(int y = 0; y < input_size; y++)
			for(int x = 0; x < input_size; x++)
				data[c * input_size * input_size + y * input_size + x] = ch[c].at<float>(y, x);
	return data;
}

struct ClassifierWrapper {
	ClassifierConfig config;
	std::vector<std::string> labels;
	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	std::string input_name;
	std::string output_name;
};

extern "C" VendDB_Classifier venddb_load_classifier(const char* config_yaml_path, int use_gpu) {
	try {
		auto wrapper    = std::make_unique<ClassifierWrapper>();
		wrapper->config = parseClassifierYAML(config_yaml_path);
		wrapper->labels = loadLabels(wrapper->config.labels_path);
		wrapper->env    = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "venddb_clf");

		Ort::SessionOptions opts;
		if(use_gpu) {
			OrtCUDAProviderOptions cuda {};
			opts.AppendExecutionProvider_CUDA(cuda);
		}
		wrapper->session = std::make_unique<Ort::Session>(*wrapper->env, wrapper->config.model_path.c_str(), opts);

		Ort::AllocatorWithDefaultOptions alloc;
		wrapper->input_name  = wrapper->session->GetInputNameAllocated(0, alloc).get();
		wrapper->output_name = wrapper->session->GetOutputNameAllocated(0, alloc).get();

		return static_cast<VendDB_Classifier>(wrapper.release());
	} catch(const std::exception& e) {
		setError(e.what());
		return nullptr;
	}
}

extern "C" void venddb_destroy_classifier(VendDB_Classifier clf) {
	delete static_cast<ClassifierWrapper*>(clf);
}

static int runClassifier(ClassifierWrapper* w, const cv::Mat& bgr, VendDB_Classification* out) {
	try {
		auto data = preprocessImage(bgr, w->config.input_size);

		auto mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		int64_t sz = w->config.input_size;
		std::array<int64_t, 4> shape { 1, 3, sz, sz };
		auto tensor = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), 4);

		const char* in       = w->input_name.c_str();
		const char* out_name = w->output_name.c_str();
		auto outputs         = w->session->Run(Ort::RunOptions { nullptr }, &in, &tensor, 1, &out_name, 1);

		float* logits = outputs[0].GetTensorMutableData<float>();
		int64_t n     = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
		int best      = (int)(std::max_element(logits, logits + n) - logits);

		// Softmax confidence for the predicted class
		float maxv = logits[best], sum = 0.0f;
		for(int i = 0; i < n; i++)
			sum += std::exp(logits[i] - maxv);
		out->confidence = 1.0f / sum;

		std::string name = (best < (int)w->labels.size()) ? w->labels[best] : "unknown";
		std::strncpy(out->class_name, name.c_str(), sizeof(out->class_name) - 1);
		out->class_name[sizeof(out->class_name) - 1] = '\0';
		return 1;
	} catch(const std::exception& e) {
		setError(e.what());
		return 0;
	}
}

extern "C" int venddb_classify_file(VendDB_Classifier clf, const char* image_path, VendDB_Classification* out) {
	auto* w     = static_cast<ClassifierWrapper*>(clf);
	cv::Mat img = cv::imread(image_path);
	if(img.empty()) {
		setError(std::string("cannot read image: ") + image_path);
		return 0;
	}
	return runClassifier(w, img, out);
}

extern "C" int venddb_classify_bytes(
	VendDB_Classifier clf, const unsigned char* data, int data_len, VendDB_Classification* out) {
	auto* w     = static_cast<ClassifierWrapper*>(clf);
	cv::Mat img = cv::imdecode(std::vector<unsigned char>(data, data + data_len), cv::IMREAD_COLOR);
	if(img.empty()) {
		setError("cannot decode image bytes");
		return 0;
	}
	return runClassifier(w, img, out);
}
