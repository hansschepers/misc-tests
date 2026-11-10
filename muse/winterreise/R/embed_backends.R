# Embedding backends -- one front door, swappable engines, incremental RDS cache.
#   embed_texts(texts, backend = "rtorch" | "api" | "ollama", model = ..., cache_dir = ...)
# returns a numeric matrix (length(texts) x d), rows L2-normalised, rownames = texts.
# Comparison notes live in embedding_backends.md (same folder as the run script).

default_model <- function(backend) {
  switch(backend
    , rtorch = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    , api    = "voyage-4-large"
    , ollama = "bge-m3"
    , stop(glue("unknown backend '{backend}'")))
}

embed_texts <- function(texts, backend = "rtorch", model = NULL, cache_dir = "cache", ...) {
  model <- model %||% default_model(backend)
  dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  cache_file <- file.path(cache_dir, glue("emb__{backend}__{gsub('[^A-Za-z0-9._-]', '_', model)}.rds"))
  cache <- if (file.exists(cache_file)) readRDS(cache_file) else NULL
  todo <- setdiff(unique(texts), rownames(cache))
  if (length(todo)) {
    t0 <- Sys.time()
    new <- switch(backend
      , rtorch = embed_rtorch(todo, repo = model, ...)
      , api    = embed_api(todo, model = model, ...)
      , ollama = embed_ollama(todo, model = model, ...)
      , stop(glue("unknown backend '{backend}'")))
    rownames(new) <- todo
    cat(glue("embed_texts[{backend}:{model}]: {length(todo)} new texts in ",
             "{round(as.numeric(difftime(Sys.time(), t0, units = 'secs')), 1)} s"), "\n")
    cache <- rbind(cache, new)
    saveRDS(cache, cache_file)
  }
  E <- cache[texts, , drop = FALSE]
  E / sqrt(rowSums(E^2))
}

# ---- 1. pure R: BERT forward pass in {torch} (libtorch, no Python) -------------------
# Ports sentence-transformers' Transformer + mean pooling for BertModel checkpoints.
# Batch size 1 => no padding, no attention mask needed.
load_bert_rtorch <- function(repo) {
  cfg <- fromJSON(hub_download(repo, "config.json"))
  stopifnot(cfg$model_type == "bert", cfg$position_embedding_type == "absolute")
  pool <- fromJSON(hub_download(repo, "1_Pooling/config.json"))
  stopifnot(isTRUE(pool$pooling_mode_mean_tokens))
  st_cfg <- tryCatch(fromJSON(hub_download(repo, "sentence_bert_config.json")), error = function(e) list())
  list(cfg = cfg
       , max_len = st_cfg$max_seq_length %||% 128L
       , w = safe_load_file(hub_download(repo, "model.safetensors"), framework = "torch")
       , tk = tokenizer$from_pretrained(repo))
}

bert_forward_rtorch <- function(ids, bert) {
  w <- bert$w
  cfg <- bert$cfg
  H <- cfg$hidden_size
  nh <- cfg$num_attention_heads
  hd <- H %/% nh
  eps <- cfg$layer_norm_eps
  L <- length(ids)
  ln <- function(x, p) nnf_layer_norm(x, H, w[[glue("{p}.weight")]], w[[glue("{p}.bias")]], eps = eps)
  lin <- function(x, p) nnf_linear(x, w[[glue("{p}.weight")]], w[[glue("{p}.bias")]])
  x <- w[["embeddings.word_embeddings.weight"]][ids + 1L, ] +
    w[["embeddings.position_embeddings.weight"]][1:L, ] +
    w[["embeddings.token_type_embeddings.weight"]][1, ]
  x <- ln(x, "embeddings.LayerNorm")
  heads <- function(t) t$reshape(c(L, nh, hd))$permute(c(2, 1, 3))
  for (l in seq_len(cfg$num_hidden_layers) - 1L) {
    p <- glue("encoder.layer.{l}")
    q <- heads(lin(x, glue("{p}.attention.self.query")))
    k <- heads(lin(x, glue("{p}.attention.self.key")))
    v <- heads(lin(x, glue("{p}.attention.self.value")))
    a <- nnf_softmax(torch_matmul(q, k$transpose(2, 3)) / sqrt(hd), dim = 3)
    ctx <- torch_matmul(a, v)$permute(c(2, 1, 3))$reshape(c(L, H))
    x <- ln(lin(ctx, glue("{p}.attention.output.dense")) + x, glue("{p}.attention.output.LayerNorm"))
    h <- nnf_gelu(lin(x, glue("{p}.intermediate.dense")))
    x <- ln(lin(h, glue("{p}.output.dense")) + x, glue("{p}.output.LayerNorm"))
  }
  as.numeric(as_array(x$mean(dim = 1)))
}

embed_rtorch <- function(texts, repo, ...) {
  bert <- load_bert_rtorch(repo)
  with_no_grad({
    out <- t(vapply(texts, function(s) {
      ids <- bert$tk$encode(s)$ids
      if (length(ids) > bert$max_len) ids <- c(ids[1:(bert$max_len - 1)], tail(ids, 1))
      bert_forward_rtorch(ids, bert)
    }, numeric(bert$cfg$hidden_size), USE.NAMES = FALSE))
  })
  out
}

# ---- 3. remote API (OpenAI-compatible /embeddings: Voyage, OpenAI, Mistral, ...) -----
# input_type is Voyage-specific ("document" | "query" | NULL); NULL drops it (OpenAI).
# Everything here is compared symmetrically (stanza vs stanza, anchor vs stanza), so
# "document" for all texts.
embed_api <- function(texts, model, base_url = "https://api.voyageai.com/v1",
                      key_env = "VOYAGE_API_KEY", input_type = "document", batch = 64, ...) {
  key <- Sys.getenv(key_env)
  if (!nzchar(key)) stop(glue("embed_api: set {key_env} (e.g. in ~/.Renviron)"))
  chunks <- split(texts, ceiling(seq_along(texts) / batch))
  do.call(rbind, lapply(chunks, function(ch) {
    body <- list(model = model, input = as.list(ch), input_type = input_type)
    resp <- request(glue("{base_url}/embeddings")) |>
      req_auth_bearer_token(key) |>
      req_body_json(body[!vapply(body, is.null, logical(1))]) |>
      req_retry(max_tries = 4) |>
      req_perform()
    d <- resp_body_json(resp, simplifyVector = TRUE)$data
    do.call(rbind, d$embedding[order(d$index)])
  }))
}

# ---- 4. local Ollama server (ollama pull bge-m3) -------------------------------------
embed_ollama <- function(texts, model, host = "http://localhost:11434", batch = 32, ...) {
  chunks <- split(texts, ceiling(seq_along(texts) / batch))
  do.call(rbind, lapply(chunks, function(ch) {
    resp <- request(glue("{host}/api/embed")) |>
      req_body_json(list(model = model, input = as.list(ch))) |>
      req_perform()
    resp_body_json(resp, simplifyVector = TRUE)$embeddings
  }))
}

cosine_sim <- function(a, b) sum(a * b) / sqrt(sum(a^2) * sum(b^2))
