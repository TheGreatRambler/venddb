#include "venddb_inference.h"

#include <fmt/core.h>

#include <cstdlib>
#include <string>

int main(int argc, char* argv[]) {
	if(argc < 3) {
		fmt::print(stderr, "Usage: {} <config.yaml> <image.jpg>\n", argv[0]);
		return 1;
	}

	const char* configPath = argv[1];
	const char* imagePath  = argv[2];

	VendDB_Detector det = venddb_load_detector(configPath, /*use_gpu=*/0);
	if(!det) {
		fmt::print(stderr, "Failed to load detector: {}\n", venddb_last_error());
		return 1;
	}

	int count              = 0;
	VendDB_Detection* dets = venddb_detect_file(det, imagePath, -1.f, -1.f, &count);
	if(!dets && count == 0) {
		const char* err = venddb_last_error();
		if(err && err[0] != '\0') {
			fmt::print(stderr, "Detection failed: {}\n", err);
			venddb_destroy_detector(det);
			return 1;
		}
	}

	fmt::print("{} detections in {}\n", count, imagePath);
	for(int i = 0; i < count; ++i) {
		fmt::print("  [{}] class={} conf={:.3f} box=({},{},{},{})\n", i, dets[i].class_name, dets[i].confidence,
			dets[i].x, dets[i].y, dets[i].width, dets[i].height);
	}

	venddb_free_detections(dets);
	venddb_destroy_detector(det);
	return 0;
}
