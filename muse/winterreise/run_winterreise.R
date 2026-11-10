# ============================================================
# Song cycles as a path through meaning-space
# (Winterreise, Dichterliebe, ... -- see cycle_config() in R/text_cycles.R)
# German text embedded, interpreted in English.
# Unit = the poet's stanza (musical repeats collapsed); song = centroid of its stanzas.
# Three maps of the same vectors:
#   A. PCA      PC1 x PC2, ribbon colour = PC3          (derived, linear, honest)
#   B. UMAP     2-D neighbourhood map, colour = PC3     (derived, nonlinear, seed-checked)
#   C. Designed anchor axes (anchors_<cycle>.csv), colour = PC3  (the reader's mental model)
# Ribbon width = within-song semantic dispersion.
# PC names: pc_labels.csv (fill in after reading axis_extremes.csv).
# Backends: embedding_backends.md. Outputs: out/<cycle>/<backend>__<model>/
# Usage: Rscript run_winterreise.R [cycle] [backend] [model]
# ============================================================

setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
args <- commandArgs(trailingOnly = TRUE)
CYCLE <- if (length(args) >= 1) args[1] else "winterreise"   # "winterreise" | "dichterliebe"
BACKEND <- if (length(args) >= 2) args[2] else "rtorch"      # "rtorch" | "api" | "ollama"
MODEL <- if (length(args) >= 3) args[3] else NULL            # NULL = default_model(BACKEND)

suppressPackageStartupMessages({
  library(data.table); library(glue); library(httr2); library(jsonlite)
  library(ggplot2); library(ggrepel); library(ggforce); library(uwot)
  if (BACKEND == "rtorch") { library(torch); library(tok); library(hfhub); library(safetensors) }
})
Sys.setenv(HF_HUB_DISABLE_SYMLINKS_WARNING = "1")
for (f in list.files("R", full.names = TRUE)) source(f, encoding = "UTF-8")
MODEL <- MODEL %||% default_model(BACKEND)
cfg <- cycle_config(CYCLE)

out_dir <- file.path("out", CYCLE, glue("{BACKEND}__{gsub('^.*/', '', MODEL)}"))
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
sink(file.path(out_dir, "run_output.txt"), split = TRUE)
cat(glue("=== {cfg$label} | {BACKEND}: {MODEL} ==="), "\n")

# 1. text ---------------------------------------------------------------
TXT <- cfg$load("cache")
stanzas <- lines_to_stanzas(TXT$lines)
titles <- stanzas[, .(title = title[1], tempo = tempo[1], n_stanza = .N), by = song]
titles[, title_en := cfg$title_en]
n_song <- nrow(titles)
cat(glue("{nrow(stanzas)} stanza units over {n_song} songs"), "\n")
print(titles[, .(song, title, title_en, tempo, n_stanza)])

# 2. embeddings ---------------------------------------------------------
embed_fn <- function(x) embed_texts(x, backend = BACKEND, model = MODEL, cache_dir = "cache")
E <- embed_fn(stanzas$text)
SV <- build_song_vectors(stanzas, E)
V <- SV$V

# 3. layouts ------------------------------------------------------------
pc <- layout_pca(V, E)
um <- layout_umap(V, E)
anchors <- fread(glue("anchors_{CYCLE}.csv"), encoding = "UTF-8")
an <- layout_anchors(V, E, anchors, embed_fn)
cat("\nPCA variance explained (PC1..3):", round(pc$var_explained, 3), "\n")
cat("UMAP seed stability (cor of 2-D distances, 5 seeds):", round(range(um$stability), 2), "\n")
cat(glue("Designed axes {an$axes[1]} x {an$axes[2]}: cos between them = {round(an$cos_between_axes, 2)}"), "\n")

songs <- titles[, .(song, title, title_en, tempo,
                    PC1 = pc$song[, 1], PC2 = pc$song[, 2], PC3 = pc$song[, 3],
                    umap1 = um$song[, 1], umap2 = um$song[, 2],
                    anchor1 = an$song[, 1], anchor2 = an$song[, 2],
                    dispersion = SV$dispersion)]
songs[, step_dist := c(1 - rowSums(V[-n_song, ] * V[-1, ]), NA)]  # distance to the NEXT song
songs[, next_title := shift(title, type = "lead")]

# 4. what do the axes mean? stanza extremes -----------------------------
ext <- rbindlist(lapply(1:3, function(j) axis_extremes(stanzas, pc$stanza[, j], glue("PC{j}"), k = 5)))
fwrite(ext, file.path(out_dir, "axis_extremes.csv"))
for (a in c("PC1", "PC2", "PC3")) {
  cat(glue("\n--- {a}: stanzas at the ends ---"), "\n")
  print(ext[axis == a, .(end, unit_id, title, score, text = substr(text, 1, 120))])
}
cat("\nSongs by PC3:\n")
print(songs[order(PC3), .(song, title, title_en, PC3 = round(PC3, 3))])

# 5. is the composer's order a 'path'? ----------------------------------
orders <- setNames(list(seq_len(n_song), TXT$alt_order), c("composer", TXT$alt_order_name))
tst <- test_order(V, orders)
cat("\nPath length (sum of consecutive cosine distances) vs 20000 random orders:\n")
print(tst[, lapply(.SD, function(x) if (is.numeric(x)) round(x, 4) else x)])
cat("\nLargest steps:\n")
print(songs[order(-step_dist)][1:5, .(song, from = title, to = next_title, step_dist = round(step_dist, 3))])
fwrite(songs, file.path(out_dir, "song_coordinates.csv"))
saveRDS(list(cycle = CYCLE, backend = BACKEND, model = MODEL, stanzas = stanzas, E = E, V = V,
             songs = songs, pc = pc, path_test = tst), file.path(out_dir, "vectors.rds"))

# 6. plots --------------------------------------------------------------
pcl <- fread("pc_labels.csv")[cycle == CYCLE & backend == BACKEND & model == MODEL]
pc_name <- function(p, sep = " / ") {
  r <- pcl[pc == p]
  if (nrow(r) == 0) return(glue("{p} (unnamed: see axis_extremes.csv)"))
  glue("{p}  (- {r$neg}{sep}+ {r$pos})")
}
pole <- function(ax, s) anchors[axis == ax & pole == s, label][1]
cap <- glue("Embedding: {BACKEND} ({MODEL}). Unit = poet's stanza; grey dots = stanzas; song = centroid.\n",
            "Ribbon colour = PC3 (sign arbitrary); width = within-song dispersion.")
mk <- function(xcol, ycol, scol, xlab, ylab, sub, file) {
  sd_dt <- data.table(x = scol[, 1], y = scol[, 2])
  p <- plot_trajectory(songs[, .(song, title, x = get(xcol), y = get(ycol), PC3, dispersion)],
                       sd_dt, xlab, ylab, glue("{cfg$label}: a path through meaning"), sub,
                       pc_name("PC3", sep = "\n"), cap)
  ggsave(file.path(out_dir, file), p, width = 13, height = 9.5, dpi = 150)
}
pv <- round(100 * pc$var_explained)
mk("PC1", "PC2", pc$stanza[, 1:2], glue("{pc_name('PC1')}  [{pv[1]} %]"), glue("{pc_name('PC2')}  [{pv[2]} %]"),
   glue("A. PCA of the {n_song} song vectors"), "A_pca_path.png")
mk("umap1", "umap2", um$stanza, "UMAP 1", "UMAP 2",
   glue("B. UMAP (cosine, n_neighbors = 8); seed stability r = {round(min(um$stability), 2)}-{round(max(um$stability), 2)}"),
   "B_umap_path.png")
mk("anchor1", "anchor2", an$stanza,
   glue("{an$axes[1]}  (- {pole(an$axes[1], '-')} / + {pole(an$axes[1], '+')})"),
   glue("{an$axes[2]}  (- {pole(an$axes[2], '-')} / + {pole(an$axes[2], '+')})"),
   glue("C. Designed axes (anchors_{CYCLE}.csv): the reader's own coordinates"), "C_designed_path.png")

long <- melt(songs[, .(song, PC3, step_dist, dispersion)], id.vars = "song")
p_diag <- ggplot(long[!is.na(value)], aes(song, value)) +
  geom_line(colour = "grey40") + geom_point(size = 2) +
  facet_wrap(~variable, ncol = 1, scales = "free_y",
             labeller = as_labeller(c(PC3 = "PC3 score", step_dist = "cosine distance to next song",
                                      dispersion = "within-song dispersion"))) +
  scale_x_continuous(breaks = songs$song, labels = paste(songs$song, songs$title)) +
  labs(x = NULL, y = NULL, title = glue("{cfg$label}: per-song diagnostics"), caption = glue("{BACKEND} ({MODEL})")) +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 50, hjust = 1), panel.grid.minor = element_blank(),
        plot.background = element_rect(fill = "white", colour = NA))
ggsave(file.path(out_dir, "D_diagnostics.png"), p_diag, width = 12, height = 8, dpi = 150)

sink()
