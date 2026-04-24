package vend

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	DRINK_EMBEDDING_DIMS  = 1536
	DRINK_COLLECTION_NAME = "drink"
)

type VendDrinkEmbed struct {
	Embedding []float32 `json:"embedding"`
}

// EmbedAskulDrink sends image bytes to the embedding service and returns the vector.
func EmbedAskulDrink(embed_url string, image []byte) (VendDrinkEmbed, error) {
	body, err := json.Marshal(struct {
		Image string `json:"image"`
	}{
		Image: base64.StdEncoding.EncodeToString(image),
	})
	if err != nil {
		return VendDrinkEmbed{}, err
	}

	resp, err := http.Post(fmt.Sprintf("%s/embed", embed_url),
		"application/json", bytes.NewReader(body))
	if err != nil {
		return VendDrinkEmbed{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		return VendDrinkEmbed{},
			fmt.Errorf("embed service returned %d: %s", resp.StatusCode, raw)
	}

	// Endpoint returns a packed little-endian float32 array.
	raw_body, err := io.ReadAll(resp.Body)
	if err != nil {
		return VendDrinkEmbed{}, err
	}
	if len(raw_body)%4 != 0 {
		return VendDrinkEmbed{},
			fmt.Errorf("embed response size %d not a multiple of 4", len(raw_body))
	}

	embedding := make([]float32, len(raw_body)/4)
	if err := binary.Read(bytes.NewReader(raw_body),
		binary.LittleEndian, embedding); err != nil {
		return VendDrinkEmbed{}, fmt.Errorf("decode embedding: %w", err)
	}

	return VendDrinkEmbed{Embedding: embedding}, nil
}

// findClosestDrink embeds the image and runs a Milvus HNSW cosine search with the given
// boolean expression (empty string = no filter). Returns the nearest VendDrink with
// ExtraJSON populated from postgres.
func (v *Vend) findClosestDrink(ctx context.Context, mc client.Client, embed_url, expr string, image []byte) (VendDrink, float32, error) {
	embed, err := EmbedAskulDrink(embed_url, image)
	if err != nil {
		return VendDrink{}, 0, fmt.Errorf("embed: %w", err)
	}

	search_vec := []entity.Vector{entity.FloatVector(embed.Embedding)}
	sp, err := entity.NewIndexHNSWSearchParam(64)
	if err != nil {
		return VendDrink{}, 0, fmt.Errorf("search param: %w", err)
	}

	results, err := mc.Search(ctx, DRINK_COLLECTION_NAME, nil, expr,
		[]string{"source_id", "code"},
		search_vec, "embedding", entity.COSINE, 1, sp)
	if err != nil {
		return VendDrink{}, 0, fmt.Errorf("milvus search: %w", err)
	}
	if len(results) == 0 || results[0].ResultCount == 0 {
		return VendDrink{}, 0, fmt.Errorf("no matches")
	}

	r := results[0]
	source_col, ok := r.Fields.GetColumn("source_id").(*entity.ColumnVarChar)
	if !ok {
		return VendDrink{}, 0, fmt.Errorf("source_id column missing or wrong type")
	}
	code_col, ok := r.Fields.GetColumn("code").(*entity.ColumnVarChar)
	if !ok {
		return VendDrink{}, 0, fmt.Errorf("code column missing or wrong type")
	}

	matched_source, err := source_col.ValueByIdx(0)
	if err != nil {
		return VendDrink{}, 0, fmt.Errorf("read source_id: %w", err)
	}
	matched_code, err := code_col.ValueByIdx(0)
	if err != nil {
		return VendDrink{}, 0, fmt.Errorf("read code: %w", err)
	}
	score := r.Scores[0]

	var extra_json []byte
	if err := v.db.QueryRow(ctx,
		`SELECT extra_json FROM drink WHERE source_id = $1 AND code = $2`,
		matched_source, matched_code,
	).Scan(&extra_json); err != nil {
		return VendDrink{}, 0, fmt.Errorf("load drink %s/%s: %w", matched_source, matched_code, err)
	}

	drink := VendDrink{Source: matched_source, Code: matched_code}
	if len(extra_json) > 0 {
		if err := json.Unmarshal(extra_json, &drink.ExtraJSON); err != nil {
			return VendDrink{}, 0, fmt.Errorf("unmarshal extra_json: %w", err)
		}
	}
	return drink, score, nil
}

// FindClosestDrinkInSource finds the nearest drink within a specific source.
func (v *Vend) FindClosestDrinkInSource(ctx context.Context, mc client.Client, embed_url, source_id string, image []byte) (VendDrink, float32, error) {
	return v.findClosestDrink(ctx, mc, embed_url,
		fmt.Sprintf(`source_id == "%s"`, source_id), image)
}

// FindClosestDrink finds the nearest drink across all sources.
func (v *Vend) FindClosestDrink(ctx context.Context, mc client.Client, embed_url string, image []byte) (VendDrink, float32, error) {
	return v.findClosestDrink(ctx, mc, embed_url, "", image)
}

func ensureDrinkCollection(ctx context.Context, mc client.Client) error {
	exists, err := mc.HasCollection(ctx, DRINK_COLLECTION_NAME)
	if err != nil {
		return fmt.Errorf("check collection: %w", err)
	}
	if exists {
		return mc.LoadCollection(ctx, DRINK_COLLECTION_NAME, false)
	}

	schema := entity.NewSchema().WithName(DRINK_COLLECTION_NAME).
		WithField(entity.NewField().WithName("id").WithDataType(entity.FieldTypeInt64).WithIsPrimaryKey(true).WithIsAutoID(true)).
		WithField(entity.NewField().WithName("source_id").WithDataType(entity.FieldTypeVarChar).WithMaxLength(36)).
		WithField(entity.NewField().WithName("code").WithDataType(entity.FieldTypeVarChar).WithMaxLength(36)).
		WithField(entity.NewField().WithName("embedding").WithDataType(entity.FieldTypeFloatVector).WithDim(DRINK_EMBEDDING_DIMS))

	if err := mc.CreateCollection(ctx, schema, entity.DefaultShardNumber); err != nil {
		return fmt.Errorf("create collection: %w", err)
	}

	idx, err := entity.NewIndexHNSW(entity.COSINE, 8, 64)
	if err != nil {
		return err
	}
	if err := mc.CreateIndex(ctx, DRINK_COLLECTION_NAME, "embedding", idx, false); err != nil {
		return fmt.Errorf("create index: %w", err)
	}
	return mc.LoadCollection(ctx, DRINK_COLLECTION_NAME, false)
}
