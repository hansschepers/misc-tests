# ============================================================
# All cycles in ONE space. The embeddings are per-text (a stanza's vector does not depend
# on which other texts are embedded), so this reuses each cycle's vectors.rds. Only the
# PROJECTION is joint:
#   raw     : PCA on all song vectors -> PCs chase BETWEEN-cycle differences ("corners")
#   centred : each cycle's mean song removed first -> PCs = shared axes of WITHIN-cycle
#             motion, so the three paths are compared on the same axes
# How cornered are the cycles?
#   R2_cycle       = share of total song variance explained by the cycle means
#   nn_same_cycle  = share of songs whose nearest neighbour (cosine) is in the same cycle
#   eta2 per PC    = share of that PC's variance explained by the cycle label
# Usage: Rscript run_joint.R [backend] [model]   (each cycle must have been run first)
# Outputs: out/joint/<backend>__<model>/
# ============================================================
setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
args <- commandArgs(trailingOnly = TRUE)
BACKEND <- if (length(args) >= 1) args[1] else "rtorch"
suppressPackageStartupMessages({ library(data.table); library(glue); library(ggplot2); library(ggrepel) })
source("R/embed_backends.R")
MODEL <- if (length(args) >= 2) args[2] else default_model(BACKEND)
CYCLES <- c(muellerin = "Die schöne Müllerin", winterreise = "Winterreise", dichterliebe = "Dichterliebe")
COLS <- c(muellerin = "#2E8B57", winterreise = "#3B6FB6", dichterliebe = "#C0392B")

run_id <- glue("{BACKEND}__{gsub('^.*/', '', MODEL)}")
out_dir <- file.path("out", "joint", run_id)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
sink(file.path(out_dir, "run_output.txt"), split = TRUE)

R <- lapply(names(CYCLES), function(cy) readRDS(file.path("out", cy, run_id, "vectors.rds")))
names(R) <- names(CYCLES)
songs <- rbindlist(lapply(names(R), function(cy) R[[cy]]$songs[, .(cycle = cy, song, title, title_en)]))
stz <- rbindlist(lapply(names(R), function(cy) R[[cy]]$stanzas[, .(cycle = cy, song, title, unit_id, text)]))
V <- do.call(rbind, lapply(R, `[[`, "V"))
E <- do.call(rbind, lapply(R, `[[`, "E"))
cat(glue("=== joint | {BACKEND}: {MODEL} | {nrow(V)} songs, {nrow(E)} stanzas ==="), "\n")

# --- how cornered? -------------------------------------------------------------------
cyc_mean <- function(M, cyc) t(sapply(names(CYCLES), function(cy) colMeans(M[cyc == cy, , drop = FALSE])))
mu <- cyc_mean(V, songs$cycle)
ss_tot <- sum(sweep(V, 2, colMeans(V))^2)
ss_between <- sum(sapply(names(CYCLES), function(cy) sum(songs$cycle == cy) * sum((mu[cy, ] - colMeans(V))^2)))
S <- V %*% t(V)
diag(S) <- -Inf
nn <- songs$cycle[apply(S, 1, which.max)]
chance <- sum(table(songs$cycle) * (table(songs$cycle) - 1)) / (nrow(V) * (nrow(V) - 1))
cat(glue("R2_cycle (variance explained by the cycle means): {round(ss_between / ss_tot, 3)}"), "\n")
cat(glue("nn_same_cycle: {round(mean(nn == songs$cycle), 3)}  (chance {round(chance, 3)})"), "\n")
cat("cosine between cycle means:\n")
print(round(mu %*% t(mu) / outer(sqrt(rowSums(mu^2)), sqrt(rowSums(mu^2))), 3))

eta2 <- function(x, g) { m <- ave(x, g); sum((m - mean(x))^2) / sum((x - mean(x))^2) }

# --- the two projections --------------------------------------------------------------
project <- function(mode) {
  if (mode == "raw") {
    Vc <- sweep(V, 2, colMeans(V))
    Ec <- sweep(E, 2, colMeans(V))
  } else {
    Vc <- V - mu[songs$cycle, ]
    Ec <- E - mu[stz$cycle, ]
  }
  pca <- prcomp(Vc, center = FALSE)
  rot <- pca$rotation[, 1:5]
  list(mode = mode, song = Vc %*% rot, stanza = Ec %*% rot,
       var = (pca$sdev^2 / sum(pca$sdev^2))[1:5],
       eta2 = apply(Vc %*% rot, 2, eta2, g = songs$cycle))
}

pcl <- fread("pc_labels.csv")
for (mode in c("raw", "centred")) {
  P <- project(mode)
  cat(glue("\n######## {mode}: variance PC1..5 = {paste(round(100 * P$var, 1), collapse = ' / ')} %"), "\n")
  cat(glue("eta2 (share of the PC explained by cycle): {paste(round(P$eta2, 2), collapse = ' / ')}"), "\n")
  for (j in 1:3) {
    d <- copy(stz)[, score := P$stanza[, j]][order(score)]
    cat(glue("\n--- {mode} PC{j}: stanza ends ---"), "\n")
    print(rbind(d[1:5][, end := "-"], d[(.N - 4):.N][, end := "+"])[
      , .(end, cycle, title, score = round(score, 3), text = substr(text, 1, 95))])
  }
  dt <- cbind(songs, data.table(x = P$song[, 1], y = P$song[, 2]))
  sd_dt <- cbind(stz[, .(cycle)], data.table(x = P$stanza[, 1], y = P$stanza[, 2]))
  lab <- dt[, .SD[c(1, .N)], by = cycle]
  pc_name <- function(p) {
    r <- pcl[cycle == glue("joint_{mode}") & backend == BACKEND & model == MODEL & pc == p]
    if (nrow(r) == 0) return(p)
    glue("{p}  (- {r$neg} / + {r$pos})")
  }
  sub <- if (mode == "raw") "PCA on all songs: axes chase what separates the cycles" else
    "Each cycle centred on its own mean: shared axes of motion WITHIN a cycle"
  p <- ggplot(dt, aes(x, y, colour = cycle)) +
    geom_point(data = sd_dt, size = 0.8, alpha = 0.25) +
    geom_path(aes(group = cycle), linewidth = 1.3, alpha = 0.75, lineend = "round",
              arrow = arrow(length = unit(2.2, "mm"), type = "closed")) +
    geom_point(shape = 21, fill = "white", size = 4.6, stroke = 0.8) +
    geom_text(aes(label = song), size = 2.3, fontface = "bold", colour = "grey15") +
    geom_text_repel(data = lab, aes(label = title), size = 3, seed = 1, box.padding = 0.6,
                    show.legend = FALSE) +
    scale_colour_manual(values = COLS, labels = CYCLES, name = NULL) +
    labs(title = glue("Three Müller/Heine cycles in one space ({mode})"), subtitle = sub,
         x = glue("{pc_name('PC1')}  [{round(100 * P$var[1])} %, cycle eta² = {round(P$eta2[1], 2)}]"),
         y = glue("{pc_name('PC2')}  [{round(100 * P$var[2])} %, cycle eta² = {round(P$eta2[2], 2)}]"),
         caption = glue("Embedding: {BACKEND} ({MODEL}). Faint dots = stanzas. Labels = first and last song of each cycle.")) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 15),
          plot.background = element_rect(fill = "white", colour = NA))
  ggsave(file.path(out_dir, glue("joint_{mode}.png")), p, width = 12, height = 9.5, dpi = 150)
  fwrite(dt[, .(cycle, song, title, title_en, PC1 = x, PC2 = y)], file.path(out_dir, glue("joint_{mode}_songs.csv")))
}
sink()
