# ============================================================
# Winterreise as a path through meaning-space
# 24 songs in Schubert's order; German text embedded, interpreted in English.
# Unit = Müller stanza (musical repeats collapsed); song = centroid of its stanzas.
# Three maps of the same vectors:
#   A. PCA      PC1 x PC2, ribbon colour = PC3          (derived, linear, honest)
#   B. UMAP     2-D neighbourhood map, colour = PC3     (derived, nonlinear, seed-checked)
#   C. Designed anchor axes from anchors.csv, colour = PC3   (the reader's mental model)
# Ribbon width = within-song semantic dispersion.
# Backend: see embedding_backends.md. Outputs: out/<backend>/
# ============================================================

setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
BACKEND <- "rtorch"   # "rtorch" | "api" | "ollama"
MODEL <- NULL         # NULL = backend default (see embed_texts())

suppressPackageStartupMessages({
  library(data.table); library(glue); library(httr2); library(jsonlite)
  library(ggplot2); library(ggrepel); library(ggforce); library(uwot)
  if (BACKEND == "rtorch") { library(torch); library(tok); library(hfhub); library(safetensors) }
})
Sys.setenv(HF_HUB_DISABLE_SYMLINKS_WARNING = "1")
for (f in list.files("R", full.names = TRUE)) source(f, encoding = "UTF-8")

out_dir <- file.path("out", BACKEND)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
sink(file.path(out_dir, "run_output.txt"), split = TRUE)

# 1. text ---------------------------------------------------------------
P <- parse_winterreise(fetch_winterreise_wikitext("cache/wikitext_raw.txt"))
stanzas <- P$stanzas
titles <- stanzas[, .(title = title[1], tempo = tempo[1], n_stanza = .N), by = song]
titles[, title_en := c("Good Night", "The Weathervane", "Frozen Tears", "Numbness",
  "The Linden Tree", "Flood", "On the River", "Looking Back", "Will-o'-the-Wisp", "Rest",
  "Dream of Spring", "Loneliness", "The Post", "The Grey Head", "The Crow", "Last Hope",
  "In the Village", "The Stormy Morning", "Delusion", "The Signpost", "The Inn", "Courage!",
  "The Mock Suns", "The Hurdy-Gurdy Man")]
cat(glue("{nrow(stanzas)} stanza units over {nrow(titles)} songs"), "\n")

# 2. embeddings ---------------------------------------------------------
embed_fn <- function(x) embed_texts(x, backend = BACKEND, model = MODEL, cache_dir = "cache")
E <- embed_fn(stanzas$text)
SV <- build_song_vectors(stanzas, E)
V <- SV$V

# 3. layouts ------------------------------------------------------------
pc <- layout_pca(V, E)
um <- layout_umap(V, E)
anchors <- fread("anchors.csv", encoding = "UTF-8")
an <- layout_anchors(V, E, anchors, embed_fn)
cat("PCA variance explained (PC1..3):", round(pc$var_explained, 3), "\n")
cat("UMAP seed stability (cor of 2-D distances, 5 seeds):", round(range(um$stability), 2), "\n")
cat(glue("Designed axes {an$axes[1]} x {an$axes[2]}: cos between them = {round(an$cos_between_axes, 2)}"), "\n")

songs <- titles[, .(song, title, title_en, tempo,
                    PC1 = pc$song[, 1], PC2 = pc$song[, 2], PC3 = pc$song[, 3],
                    umap1 = um$song[, 1], umap2 = um$song[, 2],
                    anchor1 = an$song[, 1], anchor2 = an$song[, 2],
                    dispersion = SV$dispersion)]
songs[, step_dist := c(1 - rowSums(V[-24, ] * V[-1, ]), NA)]  # distance to the NEXT song

# 4. what do the axes mean? stanza extremes -----------------------------
ext <- rbind(axis_extremes(stanzas, pc$stanza[, 1], "PC1"),
             axis_extremes(stanzas, pc$stanza[, 2], "PC2"),
             axis_extremes(stanzas, pc$stanza[, 3], "PC3"))
fwrite(ext, file.path(out_dir, "axis_extremes.csv"))
for (a in c("PC1", "PC2", "PC3")) {
  cat(glue("\n--- {a}: stanzas at the ends ---"), "\n")
  print(ext[axis == a, .(end, unit_id, title, score, text = substr(text, 1, 110))])
}
cat("\nSongs by PC3:\n")
print(songs[order(PC3), .(song, title, title_en, PC3 = round(PC3, 3))])

# 5. is Schubert's order a 'path'? --------------------------------------
orders <- list(schubert = 1:24, mueller_1824 = mueller_order_in_schubert_numbers())
tst <- test_order(V, orders)
cat("\nPath length (sum of consecutive cosine distances) vs 20000 random orders:\n")
print(tst[, lapply(.SD, function(x) if (is.numeric(x)) round(x, 4) else x)])
cat("\nLargest steps:\n")
songs[, next_title := shift(title, type = "lead")]
print(songs[order(-step_dist)][1:6, .(song, from = title, to = next_title, step_dist = round(step_dist, 3))])
fwrite(songs, file.path(out_dir, "song_coordinates.csv"))

# 6. plots --------------------------------------------------------------
# named from axis_extremes.csv (rtorch MiniLM run, 2026-11-10) -- re-check after a backend switch
PC3_LAB <- "PC3\n+ dream, illusion,\n   defiance\n- body: wounds,\n   weariness, age"
cap <- glue("Embedding: {BACKEND} ({MODEL %||% 'default model'}). Unit = Müller stanza; grey dots = stanzas; ",
            "song = centroid. Ribbon colour = PC3 (sign arbitrary); width = within-song dispersion.")
mk <- function(xcol, ycol, scol, xlab, ylab, sub, file) {
  sd_dt <- data.table(x = scol[, 1], y = scol[, 2])
  p <- plot_trajectory(songs[, .(song, title, x = get(xcol), y = get(ycol), PC3, dispersion)],
                       sd_dt, xlab, ylab, "Winterreise: a path through meaning", sub, PC3_LAB, cap)
  ggsave(file.path(out_dir, file), p, width = 13, height = 9.5, dpi = 150)
  p
}
mk("PC1", "PC2", pc$stanza[, 1:2],
   glue("PC1 ({round(100 * pc$var_explained[1])} %)"), glue("PC2 ({round(100 * pc$var_explained[2])} %)"),
   "A. PCA of the 24 song vectors", "A_pca_path.png")
mk("umap1", "umap2", um$stanza, "UMAP 1", "UMAP 2",
   glue("B. UMAP (cosine, n_neighbors = 8); seed stability r = {round(min(um$stability), 2)}-{round(max(um$stability), 2)}"),
   "B_umap_path.png")
mk("anchor1", "anchor2", an$stanza,
   glue("{an$axes[1]}  (- outer world / + inner world)"), glue("{an$axes[2]}  (- death / + life)"),
   "C. Designed axes (anchors.csv): the reader's own coordinates", "C_designed_path.png")

long <- melt(songs[, .(song, PC3, step_dist, dispersion)], id.vars = "song")
p_diag <- ggplot(long, aes(song, value)) +
  geom_line(colour = "grey40") + geom_point(size = 2) +
  facet_wrap(~variable, ncol = 1, scales = "free_y",
             labeller = as_labeller(c(PC3 = "PC3 score", step_dist = "cosine distance to next song",
                                      dispersion = "within-song dispersion"))) +
  scale_x_continuous(breaks = 1:24, labels = glue("{songs$song} {songs$title}")) +
  labs(x = NULL, y = NULL, title = "Winterreise: per-song diagnostics") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 50, hjust = 1), panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", colour = NA))
ggsave(file.path(out_dir, "D_diagnostics.png"), p_diag, width = 12, height = 8, dpi = 150)

sink()
