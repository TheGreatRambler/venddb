package vend

import (
	"fmt"

	"github.com/davidbyttow/govips/v2/vips"
)

func vipsCropToDetection(img_data []byte, det Detection) (*vips.ImageRef, error) {
	img, err := vips.NewImageFromBuffer(img_data)
	if err != nil {
		return nil, fmt.Errorf("vips decode: %w", err)
	}
	x1 := max(0, det.X)
	y1 := max(0, det.Y)
	x2 := min(img.Width(), det.X+det.Width)
	y2 := min(img.Height(), det.Y+det.Height)
	if x2 <= x1 || y2 <= y1 {
		img.Close()
		return nil, fmt.Errorf("degenerate crop (%d,%d)-(%d,%d)", x1, y1, x2, y2)
	}
	if err := img.ExtractArea(x1, y1, x2-x1, y2-y1); err != nil {
		img.Close()
		return nil, fmt.Errorf("extract area: %w", err)
	}
	return img, nil
}
