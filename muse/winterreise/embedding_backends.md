# Winterreise: embedding backends (long-term notes)

Living file: compare backends for quality, flexibility and scalability. Update the run log
below after every backend run. The code front door is `embed_texts()` in
`R/embed_backends.R`. Switch the backend at the top of `run_winterreise.R`
(`BACKEND`, `MODEL`). Outputs go to `out/<backend>/`, so runs sit side by side.

## The three routes

| | 1. rtorch (pure R) | 3. remote API | 4. Ollama (local server) |
|---|---|---|---|
| Status | **BUILT + run 2026-11-10** | code written, not run (no key yet) | code written, not run (not installed) |
| Engine | `{torch}` (libtorch 2.8 cpu) + `{tok}` + `{hfhub}` + `{safetensors}` | `httr2` POST `/embeddings` (OpenAI-compatible) | `httr2` POST `localhost:11434/api/embed` |
| Default model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, 118M) | `voyage-3.5` (key `VOYAGE_API_KEY`) | `bge-m3` (1024-d, 568M, XLM-R) |
| Python | none | none | none |
| Install | ~560 MB libtorch + 471 MB weights (HF cache, `~/.cache/huggingface`) | nothing | Ollama app (system executable) + `ollama pull bge-m3` (~1.2 GB) |
| Account | none: public, ungated HF model | API key (Voyage / OpenAI / Mistral / Cohere-compat) | none |
| Data leaves the PC | no | **yes** (fine for public-domain lyrics; NOT for customer text) | no |
| Speed seen | 82 stanzas in 10 s, CPU, batch 1 | - | - |
| Cost | free | cents per corpus | free |

## What the rtorch port covers, and what it does not

- `bert_forward_rtorch()` is a hand-written BERT encoder: embeddings, then 12 layers of
  (MHA, add+LN, GELU FFN, add+LN), then **mean pooling** over all tokens. That matches
  sentence-transformers for `model_type == "bert"` with `pooling_mode_mean_tokens`.
  The loader asserts both conditions.
- Batch size 1, so no padding and no attention mask. Truncation is at `max_seq_length`
  (128) from `sentence_bert_config.json`.
- **Verified:** semantic sanity check (`work_step1.R`). DE/EN paraphrase cos = 0.97,
  DE paraphrase = 0.71, unrelated pairs about 0.1.
- **NOT verified:** element-wise parity with Python sentence-transformers. A future check:
  compare against the HF Inference API output for the same model (needs a free HF token),
  or against reticulate once.
- Extending to other checkpoints:
  - **XLM-RoBERTa** models (`paraphrase-multilingual-mpnet-base-v2`, `multilingual-e5-*`,
    `bge-m3`): same blocks, but position ids start at `padding_idx + 1 = 2` and the
    weight names are `roberta.*`/none. That is about a 5-line change.
  - **e5** needs a `"passage: "` prefix.
  - **LaBSE** uses CLS pooling + Dense + normalize.
  - **Decoder embedders** (Qwen3-Embedding, gte-Qwen) need a different forward pass. Use
    the API or Ollama for those.

## Comparison plan (when a 2nd backend has run)

Quality is not "which map looks nicest". Use these measures:

1. **Agreement (RSA):** correlate the 24x24 song cosine-similarity matrices between
   backends (Spearman on the upper triangle). High means the backends see the same
   cycle. Also correlate the 82x82 stanza matrices.
2. **Axis stability:** does PC3 still split "body/wounds" from "dream/illusion"? Compare
   `axis_extremes.csv` across backends. Same stanzas at the ends = a robust axis.
3. **Path test:** is the Schubert path length z-score sign/size stable? (Run 1: no
   evidence of smooth drift.)
4. **Anchor sanity:** do the designed axes rank the obvious songs where a reader would
   (Frühlingstraum high on life, Wirtshaus low)? Write the expected rankings down BEFORE
   looking.
5. **Practical:** seconds per 100 stanzas, install size, privacy, cost.

Scalability: rtorch batch-1 CPU is fine to about 10^4 short texts. For more, batch with a
mask, or use Ollama/API. The incremental RDS cache (`cache/emb__<backend>__<model>.rds`)
means a text is never embedded twice per model.

## Run log

### 2026-11-10: rtorch, MiniLM-L12 multilingual, unit = Müller stanza (82)

- PCA variance PC1..3 = 13.8 / 9.5 / 9.1 %. That is a flat spectrum: 24 points in 384-d
  have no dominant axis. PC2 and PC3 are almost tied, so they could swap under another
  backend. **Watch this.**
- **PC1:** - = tears/ice/thaw (Wasserflut, Gefrorene Tränen, Erstarrung); + = encountered
  figures and their address (Die Krähe, Der Leiermann, Wetterfahne).
- **PC2:** - = seen phenomena: frost on hair, three suns, torn sky (Greise Kopf,
  Nebensonnen, Stürmischer Morgen); + = talking to one's heart ("mein Herz, mein Herz"
  in Die Post, Mut!).
- **PC3 (ribbon colour):** - = the body: wounds, faintness, age (Rast, Greise Kopf,
  Wirtshaus); + = dream, illusion, defiance (Nebensonnen, Im Dorfe, Täuschung, Mut!,
  Frühlingstraum, Gute Nacht). A literary reading: illusion against bodily reality.
  The step 23 -> 24 (Nebensonnen -> Leiermann) is the largest jump of the cycle (cos
  dist 0.59): the final illusion, then the bare figure.
- **Path test:** Schubert order path length 8.89 vs random 8.79 +- 0.27 (z = +0.36,
  p_shorter = 0.64). It is NOT a gradual drift; consecutive songs are as different as
  random pairs. Müller 1824 order: z = -0.98 (p = 0.17), smoother but not significant.
  This fits an ordering by contrast (Schubert alternates), rather than a smooth slide.
  The tangled ribbon is the finding, not a plotting failure. (The Müller order is from
  memory: verify.)
- **UMAP seed stability:** r = 0.57-0.84 across 5 seeds. That is weak, as expected with
  n = 24. Keep the PCA map (A) as the main figure; UMAP (B) is illustrative only.
- **Designed axes:** inner_outer and life_death are near-orthogonal (cos -0.09).
  Wirtshaus (21) is lowest on life (the inn = graveyard; correct). Frühlingstraum,
  Täuschung and Nebensonnen are highest (they are illusions of life; the embedding reads
  the content, not the irony). Der Leiermann is the most "outer world" song.

- **Part I vs Part II (eyeballed from `D_diagnostics.png`, not yet tested):** the mean
  step distance is about 0.34 for songs 1-12 and about 0.44 for songs 13-24. The 12 -> 13
  step (the part boundary) is also large (0.48). Part II reads as semantically jumpier.
  Test this: permute within parts.
- **Artefact:** dispersion = 0 for 19 Täuschung and 23 Die Nebensonnen. Wikisource prints
  them without stanza breaks, so each is ONE unit and gets the thinnest ribbon for a
  data reason, not a semantic one. Fix: fall back to sentence or half-stanza units
  when a song has 1 stanza.

## Ideas parked

- Unit = line or half-stanza (finer path); weight stanzas by `n_lines_sung` (musical
  emphasis).
- Smoothed drift trace (3-song running mean) over the zig-zag.
- Run the same pipeline on Die schöne Müllerin as a contrast cycle.
- Music side: join tempo/key per song (the `tempo` column is already parsed) and test
  whether semantic steps line up with key changes.
