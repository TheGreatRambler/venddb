CREATE TABLE drink (
    source_id    VARCHAR(36) NOT NULL,
    code         VARCHAR(36) NOT NULL,
    is_embedded  BOOLEAN     NOT NULL DEFAULT FALSE,
    is_exported  BOOLEAN     NOT NULL DEFAULT FALSE,
    extra_json   JSONB,
    is_embedding BOOLEAN     NOT NULL DEFAULT FALSE,
    is_exporting BOOLEAN     NOT NULL DEFAULT FALSE,
    not_a_drink  BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (source_id, code)
);

CREATE TABLE drink_image (
    source_id VARCHAR(36) NOT NULL,
    code      VARCHAR(36) NOT NULL,
    image     BYTEA       NOT NULL,
    PRIMARY KEY (source_id, code)
);

CREATE TABLE source (
    id          VARCHAR(36) PRIMARY KEY,
    updated     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_updating BOOLEAN     NOT NULL DEFAULT FALSE
);
