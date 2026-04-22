package vend

import (
	"compress/gzip"
	"compress/zlib"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
	"github.com/andybalholm/brotli"
	"github.com/davidbyttow/govips/v2/vips"
	"github.com/klauspost/compress/zstd"
	utls "github.com/refraction-networking/utls"
	"golang.org/x/net/http2"
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

type AskulDrink struct {
	Brand string `json:"brand"`

	VarCode string `json:"var_code"` // Variation code
	VarName string `json:"var_name"` // Variation name

	Code string `json:"code"` // Product code
	Name string `json:"name"`

	Description string `json:"description"`
	//ServingSize     string            `json:"serving_size"`
	//Ingredients     []string          `json:"ingredients"`
	//Allergens       []string          `json:"allergens"`
	//NutritionalInfo map[string]string `json:"nutritional_info"`
}

func decompressResponse(res *http.Response) error {
	encoding := res.Header.Get("Content-Encoding")
	var body io.ReadCloser
	var err error
	switch encoding {
	case "gzip":
		body, err = gzip.NewReader(res.Body)
		if err != nil {
			return err
		}
	case "deflate":
		body, err = zlib.NewReader(res.Body)
		if err != nil {
			return err
		}
	case "br":
		body = io.NopCloser(brotli.NewReader(res.Body))
	case "zstd":
		r, err := zstd.NewReader(res.Body)
		if err != nil {
			return err
		}
		body = r.IOReadCloser()
	default:
		return nil
	}
	res.Body = body
	res.Header.Del("Content-Encoding")
	res.ContentLength = -1
	return nil
}

type userAgentProfile struct {
	user_agent string
	hello_id   utls.ClientHelloID
}

func AskulGET(url string) (*http.Response, error) {
	USER_AGENTS := []userAgentProfile{
		// Returned 403s for me. Potentially banned.
		//{"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6723.191 Safari/537.36", utls.HelloChrome_Auto},
		//{"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.196 Safari/537.36", utls.HelloChrome_Auto},
		{"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36", utls.HelloChrome_Auto},
		{"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36", utls.HelloChrome_Auto},
		{"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.121 Safari/537.36", utls.HelloChrome_Auto},
	}

	profile := USER_AGENTS[rand.Intn(len(USER_AGENTS))]

	// Using http2 for compatability with Askul.
	transport := &http2.Transport{
		DialTLSContext: func(ctx context.Context, network, addr string, _ *tls.Config) (net.Conn, error) {
			conn, err := net.Dial(network, addr)
			if err != nil {
				return nil, err
			}

			host, _, err := net.SplitHostPort(addr)
			if err != nil {
				conn.Close()
				return nil, err
			}

			tls_conn := utls.UClient(conn, &utls.Config{
				ServerName: host,
			}, profile.hello_id)

			if err := tls_conn.HandshakeContext(ctx); err != nil {
				tls_conn.Close()
				return nil, err
			}

			return tls_conn, nil
		},
	}

	client := &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// Don't follow redirects, since this indicates the page isn't valid.
			return http.ErrUseLastResponse
		},
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", profile.user_agent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "en-US,en;q=0.9")
	req.Header.Set("Accept-Encoding", "gzip, deflate, br, zstd")
	req.Header.Set("Referer", "https://www.askul.co.jp")
	req.Header.Set("Connection", "keep-alive")
	req.Header.Set("Upgrade-Insecure-Requests", "1")
	req.Header.Set("Sec-Fetch-Dest", "document")
	req.Header.Set("Sec-Fetch-Mode", "navigate")
	req.Header.Set("Sec-Fetch-Site", "same-origin")
	req.Header.Set("Sec-Fetch-User", "?1")
	req.Header.Set("Priority", "u=0, i")
	req.Header.Set("TE", "trailers")

	res, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := decompressResponse(res); err != nil {
		res.Body.Close()
		return nil, err
	}
	return res, nil
}

func ScrapeAskulDrinks(category_prefix_urls []string) ([]AskulDrink, error) {
	//const PARAMETER_STR = "?resultType=0&sortDir=1&resultCount=100&" +
	//	"searchKeyList=000343_パッケージの種類_10500_缶" + // Can
	//	"&searchKeyList=000343_パッケージの種類_10600_カップ" + // Cup
	//	"&searchKeyList=000343_パッケージの種類_10100_ペットボトル" + // "PET Bottle"
	//	"&searchKeyList=000343_パッケージの種類_10200_瓶" + // Bottle
	//	"&searchKeyList=000343_パッケージの種類_10300_パック" // Pack (Juicebox basically)

	const PARAMETER_STR = "?resultType=0&sortDir=1&resultCount=100"

	drinks := []AskulDrink{}

	for _, category_prefix_url := range category_prefix_urls {
		page := 1
		for {
			// Wait to prevent an IP ban.
			time.Sleep(time.Second)

			// Request this page.
			url := fmt.Sprintf("%s-%d/%s", category_prefix_url, page, PARAMETER_STR)
			res, err := AskulGET(url)
			if err != nil {
				return nil, err
			}

			defer res.Body.Close()

			if res.StatusCode == 302 {
				// This page does not exist, so we have reached the end.
				// This is intended as a redirect to the first page.
				break
			}

			if res.StatusCode != 200 {
				return nil, fmt.Errorf("Askul drinks website status code error: %d %s", res.StatusCode, res.Status)
			}

			// Load the HTML document of this page.
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
					// Only one product for this item.
					drink_code, _ := s.Find("[data-catalog-item-code]").Attr("data-catalog-item-code")
					drink := AskulDrink{
						Brand: "", // TODO

						Code: drink_code,
						Name: name,

						Description: description,
					}

					drinks = append(drinks, drink)
					num_items_on_page += 1
				} else {
					var_code, _ := s.Find("[data-variation-code]").Attr("data-variation-code")

					variation_items.Each(func(_ int, v *goquery.Selection) {
						// Process each variation item
						drink_code_raw, _ := v.Find(".a-link").Attr("href")
						drink_code, _ := strings.CutPrefix(drink_code_raw, "/p/")
						drink_code, _ = strings.CutSuffix(drink_code, "/")

						drink_name, _ := v.Find(".a-tooltip_content .a-img_img").Attr("alt")

						drink := AskulDrink{
							Brand: "", // TODO

							VarCode: var_code,
							VarName: name,

							Code: drink_code,
							Name: drink_name,

							Description: description,
						}

						drinks = append(drinks, drink)
						num_items_on_page += 1
					})
				}
			})

			//fmt.Printf("Page: %d Count: %d\n URL: %s\n", page, num_items_on_page, url)

			page += 1
		}
	}

	return drinks, nil
}

const askulImageURLPattern = "https://cdn.askul.co.jp/img/product/3L1/%s_3L1.jpg"

func ExportAskulDrinks(askul_model string, drinks []AskulDrink, askul_output string) error {
	if err := os.MkdirAll(askul_output, 0755); err != nil {
		return fmt.Errorf("creating %s: %w", askul_output, err)
	}

	detector, err := LoadYOLODetector(askul_model, false)
	if err != nil {
		return fmt.Errorf("loading YOLO detector: %w", err)
	}
	defer detector.Close()

	total := len(drinks)
	for i, drink := range drinks {
		log.Printf("ExportAskulDrinks: [%d/%d] %s", i+1, total, drink.Code)

		out_path := filepath.Join(askul_output, drink.Code+".webp")
		if _, err := os.Stat(out_path); err == nil {
			continue // already processed
		}

		time.Sleep(300 * time.Millisecond)

		img_url := fmt.Sprintf(askulImageURLPattern, drink.Code)
		res, err := AskulGET(img_url)
		if err != nil {
			log.Printf("ExportAskulDrinks: %s: download: %v", drink.Code, err)
			continue
		}
		img_data, read_err := io.ReadAll(res.Body)
		res.Body.Close()
		if read_err != nil {
			log.Printf("ExportAskulDrinks: %s: read body: %v", drink.Code, read_err)
			continue
		}
		if res.StatusCode != 200 {
			log.Printf("ExportAskulDrinks: %s: HTTP %d", drink.Code, res.StatusCode)
			continue
		}

		// 80% confidence.
		dets, err := detector.DetectBytes(img_data, 0.80, -1)
		if err != nil {
			log.Printf("ExportAskulDrinks: %s: inference: %v", drink.Code, err)
			continue
		}

		// Get the best singular "drink" detection.
		var drink_dets []Detection
		for _, d := range dets {
			if d.Class == "drink" {
				drink_dets = append(drink_dets, d)
			}
		}
		if len(drink_dets) == 0 {
			log.Printf("ExportAskulDrinks: %s: no drink detected (%d total)", drink.Code, len(dets))
			continue
		}
		sort.Slice(drink_dets, func(i, j int) bool {
			return drink_dets[i].Confidence > drink_dets[j].Confidence
		})
		best := drink_dets[0]

		vips_img, err := vipsCropToDetection(img_data, best)
		if err != nil {
			log.Printf("ExportAskulDrinks: %s: crop: %v", drink.Code, err)
			continue
		}

		ep := vips.NewWebpExportParams()
		ep.Lossless = true
		webp_data, _, err := vips_img.ExportWebp(ep)
		vips_img.Close()
		if err != nil {
			log.Printf("ExportAskulDrinks: %s: webp encode: %v", drink.Code, err)
			continue
		}

		if err := os.WriteFile(out_path, webp_data, 0644); err != nil {
			log.Printf("ExportAskulDrinks: %s: save: %v", drink.Code, err)
			continue
		}

		// Output JSON metadata for this drink as well.
		json_data, err := json.Marshal(drink)
		if err != nil {
			log.Printf("ExportAskulDrinks: %s: json marshal: %v", drink.Code, err)
			continue
		}
		json_path := filepath.Join(askul_output, drink.Code+".json")
		if err := os.WriteFile(json_path, json_data, 0644); err != nil {
			log.Printf("ExportAskulDrinks: %s: json save: %v", drink.Code, err)
			continue
		}

		log.Printf("ExportAskulDrinks: saved %s (conf=%.2f, %dx%d px)",
			out_path, best.Confidence, best.Width, best.Height)
	}

	return nil
}
