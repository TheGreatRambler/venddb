#include <fmt/core.h>
#include <yolos/yolos.hpp>

int main() {
	fmt::print("Hello, World! This is a clamped int: {}\n", yolos::utils::clamp(0, 50, 100));

	yolos::det::YOLODetector detector("../yolo26_model/yolo26x.onnx", "coco.yaml", /*gpu=*/true);

	// Detect
	cv::Mat frame   = cv::imread("/Users/tgr/Downloads/PXL_20260115_033759751.jpg");
	auto detections = detector.detect(frame, /*conf=*/0.01f, /*iou=*/0.45f);

	// Process results
	for(const auto& det : detections) {
		fmt::print("Confidence: {} Box X,Y,Width,Height: {},{},{},{}\n", det.conf, det.box.x, det.box.y, det.box.width,
			det.box.height);
	}

	// Visualize
	// cv::Mat white = cv::Mat::ones(frame.size(), frame.type()) * 255;
	// cv::addWeighted(frame, 0.7, white, 0.3, 0, frame);
	detector.drawDetectionsWithMask(frame, detections);
	cv::imwrite("output.png", frame);

	return 0;
}
