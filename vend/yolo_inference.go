package vend

/*
#cgo CFLAGS:  -I${SRCDIR}/../inference/include
#cgo LDFLAGS: -L${SRCDIR}/../inference/build -lvenddb_inference
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../inference/build
#cgo pkg-config: libonnxruntime opencv4
#include "venddb_inference.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Detection is a single YOLO bounding-box result.
type Detection struct {
	Class      string
	Confidence float32
	X, Y       int
	Width      int
	Height     int
}

// YOLODetector wraps a loaded venddb_inference detector.
type YOLODetector struct {
	handle C.VendDB_Detector
}

// LoadYOLODetector loads a detector from the given YAML config path.
// The YAML must contain model_path, conf_threshold, iou_threshold, and a classes list.
func LoadYOLODetector(config_yaml_path string, use_gpu bool) (*YOLODetector, error) {
	c_path := C.CString(config_yaml_path)
	defer C.free(unsafe.Pointer(c_path))

	gpu := C.int(0)
	if use_gpu {
		gpu = 1
	}

	h := C.venddb_load_detector(c_path, gpu)
	if h == nil {
		return nil, fmt.Errorf("venddb_load_detector: %s", C.GoString(C.venddb_last_error()))
	}
	return &YOLODetector{handle: h}, nil
}

// Close releases the native detector resources.
func (d *YOLODetector) Close() {
	if d.handle != nil {
		C.venddb_destroy_detector(d.handle)
		d.handle = nil
	}
}

// DetectFile runs inference on an image file.
// Pass conf/iou < 0 to use the thresholds from the config YAML.
func (d *YOLODetector) DetectFile(image_path string, conf, iou float32) ([]Detection, error) {
	c_path := C.CString(image_path)
	defer C.free(unsafe.Pointer(c_path))

	var count C.int
	raw := C.venddb_detect_file(d.handle, c_path, C.float(conf), C.float(iou), &count)
	if raw == nil && count == 0 {
		if err_msg := C.GoString(C.venddb_last_error()); err_msg != "" {
			return nil, fmt.Errorf("venddb_detect_file: %s", err_msg)
		}
		return nil, nil
	}
	defer C.venddb_free_detections(raw)

	return cDetsToGo(raw, int(count)), nil
}

// DetectBytes runs inference on raw image bytes (JPEG, PNG, etc.).
// Supports all images supported by `cv::imdecode`.
// Pass conf/iou < 0 to use the thresholds from the config YAML.
func (d *YOLODetector) DetectBytes(data []byte, conf, iou float32) ([]Detection, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("empty image data")
	}

	var count C.int
	raw := C.venddb_detect_bytes(
		d.handle,
		(*C.uchar)(unsafe.Pointer(&data[0])),
		C.int(len(data)),
		C.float(conf),
		C.float(iou),
		&count,
	)
	if raw == nil && count == 0 {
		if err_msg := C.GoString(C.venddb_last_error()); err_msg != "" {
			return nil, fmt.Errorf("venddb_detect_bytes: %s", err_msg)
		}
		return nil, nil
	}
	defer C.venddb_free_detections(raw)

	return cDetsToGo(raw, int(count)), nil
}

func cDetsToGo(raw *C.VendDB_Detection, n int) []Detection {
	out := make([]Detection, n)
	for i, d := range unsafe.Slice(raw, n) {
		out[i] = Detection{
			Class:      C.GoString(&d.class_name[0]),
			Confidence: float32(d.confidence),
			X:          int(d.x),
			Y:          int(d.y),
			Width:      int(d.width),
			Height:     int(d.height),
		}
	}
	return out
}
