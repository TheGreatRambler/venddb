#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// YOLO26 Detector
// ============================================================================

// Single detection result.
typedef struct VendDB_Detection {
	char class_name[128];
	float confidence;
	int x, y, width, height; // pixel-space bounding box
} VendDB_Detection;

// Opaque handle to a loaded YOLO detector.
typedef void* VendDB_Detector;

// Load a detector from a YAML config file.
// The YAML must contain: model_path, conf_threshold, iou_threshold, classes list.
// model_path is resolved relative to the YAML file's directory if not absolute.
// use_gpu: 0 = CPU, 1 = GPU (CUDA).
// Returns NULL on failure; call venddb_last_error() for the message.
VendDB_Detector venddb_load_detector(const char* config_yaml_path, int use_gpu);

// Free a detector created by venddb_load_detector.
void venddb_destroy_detector(VendDB_Detector detector);

// Run detection on an image file.
// conf_threshold / iou_threshold override the values from the config (pass -1 to use config defaults).
// On success, returns a heap-allocated array of *out_count detections.
// The caller must free the array with venddb_free_detections().
// Returns NULL on failure; call venddb_last_error() for the message.
VendDB_Detection* venddb_detect_file(
	VendDB_Detector detector, const char* image_path, float conf_threshold, float iou_threshold, int* out_count);

// Run detection on raw image bytes (any format OpenCV can decode: JPEG, PNG, …).
// Same semantics as venddb_detect_file.
VendDB_Detection* venddb_detect_bytes(VendDB_Detector detector, const unsigned char* data, int data_len,
	float conf_threshold, float iou_threshold, int* out_count);

// Free the detection array returned by venddb_detect_file / venddb_detect_bytes.
void venddb_free_detections(VendDB_Detection* detections);

// Return the last error message for the calling thread (never NULL).
const char* venddb_last_error(void);

// ============================================================================
// ConvNeXtV2 classifier
// ============================================================================

typedef struct VendDB_Classification {
	char class_name[128];
	float confidence; // softmax probability of the predicted class
} VendDB_Classification;

// Opaque handle to a loaded ConvNeXtV2 classifier.
typedef void* VendDB_Classifier;

// Load a classifier from a YAML config file.
// The YAML must contain: model_path (.onnx), labels_path (.json), input_size (int).
// Paths are resolved relative to the YAML file's directory if not absolute.
// use_gpu: 0 = CPU, 1 = CUDA.
// Returns NULL on failure; call venddb_last_error() for the message.
VendDB_Classifier venddb_load_classifier(const char* config_yaml_path, int use_gpu);

// Free a classifier created by venddb_load_classifier.
void venddb_destroy_classifier(VendDB_Classifier classifier);

// Classify an image file. Fills *out and returns 1 on success, 0 on failure.
int venddb_classify_file(VendDB_Classifier classifier, const char* image_path, VendDB_Classification* out);

// Classify raw image bytes (any format OpenCV can decode).
int venddb_classify_bytes(
	VendDB_Classifier classifier, const unsigned char* data, int data_len, VendDB_Classification* out);

#ifdef __cplusplus
}
#endif
