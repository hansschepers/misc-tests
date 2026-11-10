# Song cycles: embedding backends (long-term notes)

Living file: compare backends for quality, flexibility and scalability. Update the run log
below after every backend run. The code front door is `embed_texts()` in
`R/embed_backends.R`.

- Run: `Rscript run_winterreise.R <cycle> <backend> [model]`. The cycles are
  `winterreise` and `dichterliebe` (configs in `R/text_cycles.R`); the backends are
  `rtorch`, `api` and `ollama`.
- Outputs: `out/<cycle>/<backend>__<model>/`.
- Compare: `Rscript compare_backends.R <cycle>` writes
  `out/<cycle>/backend_comparison.txt`.
- PC names: `pc_labels.csv`, one row per cycle x backend x model x PC. Fill these in after
  reading `axis_extremes.csv`.
- Designed axes: `anchors_<cycle>.csv`.

## The three routes

| | 1. rtorch (pure R) | 3. remote API | 4. Ollama (local server) |
|---|---|---|---|
| Status | **BUILT + run 2026-11-10** | **run 2026-11-10 (Voyage)** | code written, not run (not installed) |
| Engine | `{torch}` (libtorch 2.8 cpu) + `{tok}` + `{hfhub}` + `{safetensors}` | `httr2` POST `/embeddings` (OpenAI-compatible; Voyage `input_type = "document"`) | `httr2` POST `localhost:11434/api/embed` |
| Default model | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-d, 118M) | `voyage-4-large` (1024-d; key `VOYAGE_API_KEY` in env) | `bge-m3` (1024-d, 568M, XLM-R) |
| Python | none | none | none |
| Install | ~560 MB libtorch + 471 MB weights (HF cache, `~/.cache/huggingface`) | nothing | Ollama app (system executable) + `ollama pull bge-m3` (~1.2 GB) |
| Account | none: public, ungated HF model | API key (Voyage / OpenAI / Mistral / Cohere-compat) | none |
| Data leaves the PC | no | **yes** (fine for public-domain lyrics; NOT for customer text) | no |
| Speed seen | 82 stanzas in 10 s, CPU, batch 1 | 82 stanzas in 2-21 s (first call slow) | - |
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

### 2026-11-10: Voyage (voyage-4-large) on both cycles; Dichterliebe added

Dichterliebe text: Heine, *Buch der Lieder* (1827), one de.wikisource page per poem
(Heine's text and spelling, e.g. "Thränen" and "seyn", not Schumann's sung text). The
"poet's order" is the position in the *Lyrisches Intermezzo* list: Heine numbers 1 2 3 4
7 11 18 22 20 41 40 46 56 57 44 66. Schumann's order is nearly Heine's (he swaps 8/9 and
10/11 and moves "Aus alten Märchen" late).

**Cross-backend agreement (`out/<cycle>/backend_comparison.txt`):**

| | Winterreise | Dichterliebe |
|---|---|---|
| RSA songs (Spearman of cosine matrices) | 0.61 | 0.75 |
| RSA stanzas | 0.50 | 0.74 |
| step-distance profile cor | **0.72** | **0.80** |
| PC1..3 variance (rtorch / voyage) | 14/10/9 % vs 11/9/7 % | 18/16/12 % vs 14/13/12 % |
| PC correspondence | **NOT 1:1**. rtorch PC3 ~ -voyage PC2 (r = -0.65) | PC1 and PC2 swapped (0.77 / 0.68); **PC3 = PC3 (0.67)** |
| UMAP seed stability (rtorch / voyage) | 0.57-0.84 / 0.52-0.89 | 0.77-0.89 / **0.14-0.67** |

**What survives a backend switch (trust it):**

- **The path tests.**
  - Winterreise, Schubert's order: z = +0.36 / +0.63, NOT smoother than random.
  - Winterreise, Müller's order: z = -0.98 / -0.92, smoother but not significant.
  - Dichterliebe: Schumann's order z = -0.76 / -0.79 and Heine's order z = -0.74 / -0.53,
    both mildly smoother, not significant. Dichterliebe reads more like a continuous
    story than Winterreise.
- **The step profile** (r = 0.72 / 0.80):
  - Winterreise: Nebensonnen -> Leiermann is a top-4 jump in both.
  - Dichterliebe: the largest jumps sit around "Ein Jüngling liebt ein Mädchen" (10 -> 11
    -> 12) in both.
- **Dichterliebe PC3:** the story told in the third person, with irony ("Ein Jüngling
  liebt ein Mädchen", "Die Rose, die Lilie") against the enchanted garden and fairy tale
  ("Aus alten Märchen", "Am leuchtenden Sommermorgen"). The same stanzas sit at the ends
  in both backends.
- **The illusion-vs-reality axis in Winterreise** exists in both backends, but at
  different ranks: rtorch PC3 (body/wounds vs dream/illusion) ~ voyage PC2 (illusion and
  defiance: Nebensonnen, Mut!, Täuschung vs omens: Wirtshaus, Krähe, Wetterfahne). So the
  CONTENT is robust; "it is PC3" is not.

**What does NOT survive:**

- PC numbering in Winterreise (24 songs, flat spectrum).
- UMAP layouts, especially Voyage on 16 songs (r down to 0.14). UMAP is decoration here.

**Lesson for the plots:** colouring by "PC3" is backend-dependent. Better: colour by a
NAMED axis. Use either (a) a designed anchor axis, or (b) the PC from run X that best
matches a reference axis (Procrustes-align PCs to the rtorch run).

**rtorch vs Voyage at this scale:** Voyage spreads stanzas more evenly (lower variance per
PC, smaller cosine distances overall: path lengths 6.9 vs 8.9). The MiniLM PCs are
"peakier", so easier to name but possibly overfitting to wording. Neither is "right"; the
agreement table is the quality signal.

### 2026-11-10: Die schöne Müllerin added; joint space of all three cycles

**Text:** Müller, Hatfield 1906 critical edition on de.wikisource, one page per poem.
Schubert's 20 of Müller's 25, in Müller's order, so there is no alternative order to
test. The wikitext is full of `{{Zeile|5}}` line counters and `{{Seite|..}}` page markers,
which are stripped. `'''sie'''` (spaced-out emphasis) becomes plain text. `poem_lines()`
now STOPS if any markup or a bare number survives.

Wikimedia rate limit: two parallel runs fetching 20 pages each got HTTP 429. The fetcher
is now throttled to 1 request/s and retries; it fetches only once (cached).

**Path test: Müllerin is the only significantly smooth cycle.**

| cycle | z (rtorch) | z (voyage) | reading |
|---|---|---|---|
| Die schöne Müllerin | **-2.65 (p = 0.007)** | **-3.66 (p = 0.0002)** | a narrative: each song follows from the last |
| Dichterliebe | -0.76 | -0.79 | mildly continuous |
| Winterreise | +0.36 | +0.63 | ordered by contrast, not drift |

**Joint space (`run_joint.R <backend>`, outputs in `out/joint/<run>/`):**

- The embeddings do not change when the cycles are embedded together; each text is
  encoded on its own. Only the PROJECTION is joint.
- How cornered are the cycles? The cycle means are very similar (cosine 0.88-0.95), and
  the cycle explains only 10-12 % of song variance. But **68-72 % of songs have their
  nearest neighbour in their own cycle** (chance 33 %). Each cycle has its own
  neighbourhood inside a shared space.
- **raw PCA:** PC1 is mostly the cycle (eta² = 0.46 rtorch / 0.68 voyage). One end is
  Winterreise (cold, weariness, grave, grey head); the other is the flowers and May of
  Dichterliebe and Müllerin. PC2 picks out Müllerin's hunter/jealousy episode (14-15)
  as its own corner. So **yes: corners**, and the first axes are spent on telling the
  cycles apart.
  - Also seen: Winterreise's opening (Gute Nacht) sits next to Müllerin's opening (Das
    Wandern). Both begin with wandering.
- **centred PCA** (each cycle minus its own mean; eta² = 0 by construction) gives
  shared axes of motion WITHIN a cycle. In both backends, **Der Leiermann and Die alten,
  bösen Lieder (the two closing songs) share an extreme pole.** rtorch calls it "the
  outsider"; voyage calls it "loud outward spectacle" (coffin, wedding music, hunter,
  hurdy-gurdy man) against "intimate inner longing". Both cycles end by turning outward,
  towards a figure or an object, away from the self.

## Ideas parked

- Colour the ribbon by a NAMED axis (anchor axis, or PCs aligned to a reference run)
  instead of raw PC3; see the lesson above.
- One joint map with Winterreise and Dichterliebe embedded together (same PCA), to show
  where the two cycles overlap.
- Heine-number check: Wikipedia says "Es leuchtet meine Liebe" is Heine no. 46; our 1827
  list gives 47 (1827 has "Im wunderschönen Monat Mai" as no. 1, which the 1823 edition
  lacks?). The order is unaffected.

- Unit = line or half-stanza (finer path); weight stanzas by `n_lines_sung` (musical
  emphasis).
- Smoothed drift trace (3-song running mean) over the zig-zag.
- Run the same pipeline on Die schöne Müllerin as a contrast cycle.
- Music side: join tempo/key per song (the `tempo` column is already parsed) and test
  whether semantic steps line up with key changes.
