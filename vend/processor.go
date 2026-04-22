package vend

import (
	"fmt"
	"log"

	"github.com/davidbyttow/govips/v2/vips"
)

var imageExtensions = map[string]bool{
	".jpg":  true,
	".jpeg": true,
	".png":  true,
	".webp": true,
}

type VendImageMeta struct {
	Lat float64 `json:"lat"`
	Lng float64 `json:"lng"`
}

type VendImageItem struct {
	Type     string `json:"type"`     // Corresponding to unique ID of Askul item in database
	Price    int    `json:"price"`    // Number of hundreths of units of currency
	Currency string `json:"currency"` // ISO 4217 currency code
}

type VendImageResult struct {
	BrandType string          `json:"brand_type"` // Corresponding to unique ID of vending machine company in database
	AdminCode string          `json:"admin_code"` // Also known as "kanri bangō", or "administration number". Isn't guaranteed to be a number
	Items     []VendImageItem `json:"items"`
}

func (s *VendServer) ProcessImage(image []byte, meta VendImageMeta) (VendImageResult, error) {
	image_vips, err := vips.NewImageFromBuffer(image)
	if err != nil {
		return VendImageResult{}, fmt.Errorf("vips decode: %w", err)
	}
	defer image_vips.Close()

	// Check image dimensions
	if image_vips.Width() > 100000 || image_vips.Height() > 100000 {
		return VendImageResult{}, fmt.Errorf("image dimensions exceed maximum: %dx%d", image_vips.Width(), image_vips.Height())
	}

	detections, err := s.vend_detector.DetectBytes(image, 0.20, -1)
	if err != nil {
		// Attempt reencode into webp if YOLO's cv::imdecode can't handle the input.
		ep := vips.NewWebpExportParams()
		ep.Lossless = true
		image_webp, _, err := image_vips.ExportWebp(ep)
		if err != nil {
			return VendImageResult{}, fmt.Errorf("vips webp export: %w", err)
		}

		detections, err = s.vend_detector.DetectBytes(image_webp, 0.20, -1)
		if err != nil {
			return VendImageResult{}, fmt.Errorf("detection failed after webp reencode: %w", err)
		}
	}

	for _, det := range detections {
		log.Printf("Detection: class=%s conf=%.2f box=[x=%d,y=%d,width=%d,height=%d]", det.Class, det.Confidence, det.X, det.Y, det.Width, det.Height)
	}

	_ = detections // TODO: process detections into VendImageResult

	/*
		if s.images_dir == "" {
			log.Println("Image processor: no images directory configured, skipping")
			return
		}

		if api_key == "" {
			log.Println("Image processor: OpenRouter API key not set, skipping")
			return
		}

		output_dir := s.output_dir
		if output_dir == "" {
			output_dir = s.images_dir + "_processed"
		}

		if err := os.MkdirAll(output_dir, 0755); err != nil {
			log.Printf("Image processor: failed to create output dir %s: %v", output_dir, err)
			return
		}

		entries, err := os.ReadDir(s.images_dir)
		if err != nil {
			log.Printf("Image processor: failed to read images dir %s: %v", s.images_dir, err)
			return
		}

		client := NewOpenRouterClient(api_key)

		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}

			name := entry.Name()
			ext := strings.ToLower(filepath.Ext(name))
			if !imageExtensions[ext] {
				continue
			}

			base_name := strings.TrimSuffix(name, filepath.Ext(name))
			output_path := filepath.Join(output_dir, base_name+".json")

			if _, err := os.Stat(output_path); err == nil {
				log.Printf("Image processor: skipping %s (already processed)", name)
				continue
			}

			image_path := filepath.Join(s.images_dir, name)
			log.Printf("Image processor: processing %s", name)

			result, err := client.ProcessImage(image_path)
			if err != nil {
				log.Printf("Image processor: error processing %s: %v", name, err)
				continue
			}

			json_data, err := json.MarshalIndent(result, "", "  ")
			if err != nil {
				log.Printf("Image processor: error marshaling result for %s: %v", name, err)
				continue
			}

			if err := os.WriteFile(output_path, json_data, 0644); err != nil {
				log.Printf("Image processor: error writing %s: %v", output_path, err)
				continue
			}

			log.Printf("Image processor: saved %s", output_path)

			annotated_path := filepath.Join(output_dir, name)
			if err := drawAnnotatedImage(image_path, annotated_path, result.Items); err != nil {
				log.Printf("Image processor: error drawing annotations for %s: %v", name, err)
			} else {
				log.Printf("Image processor: saved annotated %s", annotated_path)
			}
		}

		log.Println("Image processor: all images processed")
	*/

	return VendImageResult{}, nil
}
