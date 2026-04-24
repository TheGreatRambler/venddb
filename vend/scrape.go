package vend

import (
	"fmt"
	"io"
	"log"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/davidbyttow/govips/v2/vips"
	"github.com/imroc/req/v3"
)

/*
// splitIngredients splits a Japanese ingredients string by 、 (ideographic comma)
// but only when not inside parentheses or brackets (half-width or full-width).
func splitIngredients(s string) []string {
	var results []string
	depth := 0
	start := 0
	runes := []rune(s)
	for i, r := range runes {
		switch r {
		case '(', '[', '（', '［':
			depth++
		case ')', ']', '）', '］':
			depth--
		case '、':
			if depth == 0 {
				results = append(results, strings.TrimSpace(string(runes[start:i])))
				start = i + 1
			}
		}
	}
	if start < len(runes) {
		results = append(results, strings.TrimSpace(string(runes[start:])))
	}
	return results
}

type CocaColaBrand struct {
	Name    string `json:"name"`
	NameENG string `json:"name_eng"`
	URL     string `json:"url"`
}

type CocaColaDrink struct {
	Brand           *CocaColaBrand    `json:"brand"`
	Title           string            `json:"title"`
	Description     string            `json:"description"`
	ImageURL        string            `json:"image_url"`
	ServingSize     string            `json:"serving_size"`
	Ingredients     []string          `json:"ingredients"`
	Allergens       []string          `json:"allergens"`
	NutritionalInfo map[string]string `json:"nutritional_info"`
}

func (s *VendServer) ScrapeCocaColaDrinks() ([]CocaColaBrand, []CocaColaDrink, error) {
	// Request all drinks from Coca-Cola Japan's website.
	res, err := http.Get("https://www.coca-cola.com/jp/ja/brands")
	if err != nil {
		return nil, nil, err
	}

	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, nil, fmt.Errorf("Coca Cola drinks website status code error: %d %s", res.StatusCode, res.Status)
	}

	// Load the HTML document
	main_doc, err := goquery.NewDocumentFromReader(res.Body)
	if err != nil {
		return nil, nil, err
	}

	brands := []CocaColaBrand{}
	main_doc.Find(".brands-list .cmp-image__link").Each(func(_ int, s *goquery.Selection) {
		link, exists := s.Attr("href")
		if !exists {
			return
		}

		name, exists := s.Find("img").Attr("alt")
		if !exists {
			return
		}

		brands = append(brands, CocaColaBrand{
			Name:    name,
			NameENG: link[strings.LastIndex(link, "/")+1:],
			URL:     fmt.Sprintf("https://www.coca-cola.com%s", link),
		})
	})

	drinks := []CocaColaDrink{}
	for _, brand := range brands {
		fmt.Printf("Scraping brand: %s\n", brand.NameENG)

		res, err := http.Get(fmt.Sprintf("%s/products", brand.URL))
		if err != nil {
			return nil, nil, err
		}
		defer res.Body.Close()

		var doc *goquery.Document
		if res.StatusCode != 200 {
			// Products are on homepage.
			res, err := http.Get(brand.URL)
			if err != nil {
				return nil, nil, err
			}
			defer res.Body.Close()

			if res.StatusCode != 200 {
				return nil, nil, fmt.Errorf("%s brand page status code error: %d %s", brand.Name, res.StatusCode, res.Status)
			}

			// Read this page.
			doc, err = goquery.NewDocumentFromReader(res.Body)
			if err != nil {
				return nil, nil, err
			}
		} else {
			// Read this page.
			doc, err = goquery.NewDocumentFromReader(res.Body)
			if err != nil {
				return nil, nil, err
			}
		}

		doc.Find(".cmp-container .product-information").Each(func(_ int, s *goquery.Selection) {
			drink_info := CocaColaDrink{}
			drink_info.NutritionalInfo = make(map[string]string)

			var exists bool
			drink_info.ImageURL, exists = s.Find("img").Attr("src")
			if !exists {
				return
			}

			drink_info.ImageURL = fmt.Sprintf("https://www.coca-cola.com%s", drink_info.ImageURL)

			product_information := s.Find(".product-information__content")

			drink_info.Title = product_information.Find(".cmp-title__text").Text()
			drink_info.Description = strings.TrimSpace(product_information.Find(".cmp-text").Text())
			if drink_info.Title == "" {
				return
			}

			s.Find(".nutrition .nutritional-information .nutritional-information__row.column-3--bold:not(.row--header)").Each(func(_ int, s *goquery.Selection) {
				col1 := strings.TrimSpace(s.Find(".column1").Text())
				col2 := strings.TrimSpace(s.Find(".column2").Text())
				col3 := strings.TrimSpace(s.Find(".column3").Text())

				if strings.HasSuffix(col1, "当たり") {
					// Serving size.
					drink_info.ServingSize, _ = strings.CutSuffix(col1, "当たり")
				} else if strings.HasPrefix(col2, "原材料名") {
					// Ingredients and allergens.
					for line := range strings.SplitSeq(col2, "\n") {
						if strings.HasPrefix(line, "アレルギー特定原材料：") {
							allergens_string_raw, _ := strings.CutPrefix(line, "アレルギー特定原材料：")
							drink_info.Allergens = splitIngredients(allergens_string_raw)
						} else if !strings.HasPrefix(line, "原材料名") {
							// List of ingredients.
							drink_info.Ingredients = splitIngredients(line)
						}
					}
				} else {
					// Standard nutritional info.
					nutrient := col1
					amount := col3
					drink_info.NutritionalInfo[nutrient] = amount
				}

			})

			drink_info.Brand = &brand
			drinks = append(drinks, drink_info)
		})
	}

	return brands, drinks, nil
}
*/

// https://www.useragentstring.com/pages/Chrome/ for version 120
var askulClient = req.NewClient().
	ImpersonateChrome().
	SetUserAgent("Mozilla/5.0 (Windows Server 2012 R2 Standard; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5975.80 Safari/537.36")

func ScrapeAskulDrinks(category_prefix_urls []string) ([]VendDrink, error) {
	const PARAMETER_STR = "?resultType=0&sortDir=1&resultCount=100"

	var drinks []VendDrink

	for _, category_prefix_url := range category_prefix_urls {
		page := 1
		for {
			// Wait to prevent an IP ban.
			time.Sleep(time.Second)

			// Disallow redirects while scraping (indicates a non existent page).
			url := fmt.Sprintf("%s-%d/%s", category_prefix_url, page, PARAMETER_STR)
			res, err := askulClient.
				SetRedirectPolicy(req.NoRedirectPolicy()).
				R().
				SetHeader("Referer", "https://www.askul.co.jp").
				Get(url)
			if err != nil {
				return nil, err
			}

			defer res.Body.Close()

			if res.StatusCode == 302 {
				// This page does not exist, so we have reached the end.
				break
			}

			if res.StatusCode != 200 {
				return nil, fmt.Errorf("Askul drinks website status code error: %d %s from %s", res.StatusCode, res.Status, url)
			}

			page_doc, err := goquery.NewDocumentFromReader(res.Body)
			if err != nil {
				return nil, err
			}

			num_items_on_page := 0
			page_doc.Find("#itemRefineSearch-searchResultBody .o-itemdescription").Each(func(_ int, s *goquery.Selection) {
				name := strings.TrimSpace(s.Find(".o-productdetailinfo .a-heading_text").Text())
				description := strings.TrimSpace(s.Find(".o-productdetailinfo .u-maxline").Text())
				variation_items := s.Find(".m-thumbnailimages_item")

				if variation_items.Length() == 0 {
					drink_code, _ := s.Find("[data-catalog-item-code]").Attr("data-catalog-item-code")
					drinks = append(drinks, VendDrink{
						Source: "askul",
						Code:   drink_code,
						ExtraJSON: map[string]any{
							"name":        name,
							"description": description,
						},
					})
					num_items_on_page++
				} else {
					var_code, _ := s.Find("[data-variation-code]").Attr("data-variation-code")

					variation_items.Each(func(_ int, v *goquery.Selection) {
						drink_code_raw, _ := v.Find(".a-link").Attr("href")
						drink_code, _ := strings.CutPrefix(drink_code_raw, "/p/")
						drink_code, _ = strings.CutSuffix(drink_code, "/")
						drink_name, _ := v.Find(".a-tooltip_content .a-img_img").Attr("alt")

						drinks = append(drinks, VendDrink{
							Source: "askul",
							Code:   drink_code,
							ExtraJSON: map[string]any{
								"var_code":    var_code,
								"var_name":    name,
								"name":        drink_name,
								"description": description,
							},
						})
						num_items_on_page++
					})
				}
			})

			page++
			_ = num_items_on_page
		}
	}

	return drinks, nil
}

const ASKUL_IMAGE_URL = "https://cdn.askul.co.jp/img/product/3L1/%s_3L1.jpg"

// VendDrink is a normalised drink record from any source.
type VendDrink struct {
	Source    string
	Code      string
	ExtraJSON map[string]any
}

// VendDrinkExport holds the processed output for a single drink.
type VendDrinkExport struct {
	Image []byte
}

// ExportAskulDrink downloads the Askul CDN image for drink, runs YOLO detection,
// crops to the best bounding box, and returns a VendDrinkExport.
func ExportAskulDrink(detector *YOLODetector, drink VendDrink) (VendDrinkExport, bool, error) {
	// Do not allow for redirects, as that just redirects to the noimage default.
	img_url := fmt.Sprintf(ASKUL_IMAGE_URL, drink.Code)
	res, err := askulClient.
		SetRedirectPolicy(req.NoRedirectPolicy()).
		R().
		SetHeader("Referer", "https://www.askul.co.jp").
		Get(img_url)
	if err != nil {
		return VendDrinkExport{}, false, fmt.Errorf("download: %w", err)
	}
	img_data, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return VendDrinkExport{}, false, fmt.Errorf("read body: %w", err)
	}

	// A redirect is just to `noimage`, which indicates the drink might not exist anymore.
	// Basically the way 404s are represented by this service.
	// TODO use a different flag than `not_a_drink` to indicate this.`
	if res.StatusCode == 302 {
		return VendDrinkExport{}, true, fmt.Errorf("redirect to noimage")
	}

	if res.StatusCode != 200 {
		return VendDrinkExport{}, false, fmt.Errorf("HTTP %d", res.StatusCode)
	}

	dets, err := detector.DetectBytes(img_data, 0.90, -1)
	if err != nil {
		return VendDrinkExport{}, false, fmt.Errorf("inference: %w", err)
	}

	var drink_dets []Detection
	for _, d := range dets {
		if d.Class == "drink" {
			drink_dets = append(drink_dets, d)
		}
	}
	if len(drink_dets) == 0 {
		return VendDrinkExport{}, true, fmt.Errorf("no drink detected (%d total detections)", len(dets))
	}

	// Get the best singular drink detection.
	var best Detection
	var best_confidence = float32(0.0)
	for _, drink_det := range drink_dets {
		if drink_det.Confidence > best_confidence {
			best = drink_det
			best_confidence = drink_det.Confidence
		}
	}

	vips_img, err := vipsCropToDetection(img_data, best)
	if err != nil {
		return VendDrinkExport{}, false, fmt.Errorf("crop: %w", err)
	}

	ep := vips.NewWebpExportParams()
	ep.Lossless = true
	webp_data, _, err := vips_img.ExportWebp(ep)
	vips_img.Close()
	if err != nil {
		return VendDrinkExport{}, false, fmt.Errorf("webp encode: %w", err)
	}

	log.Printf("ExportAskulDrink: %s conf=%.2f %dx%d px %vkb", drink.Code, best.Confidence, best.Width, best.Height, len(webp_data)/1024)
	return VendDrinkExport{Image: webp_data}, true, nil
}
