# Semantic trajectory: song vectors, three 2-D layouts (PCA / UMAP / designed anchor axes),
# path statistics, PC interpretation, plotting.

# song vector = normalised centroid of its stanza embeddings; dispersion = mean (1 - cos) to it
build_song_vectors <- function(stanzas, E) {
  songs <- sort(unique(stanzas$song))
  V <- t(vapply(songs, function(i) colMeans(E[stanzas$song == i, , drop = FALSE]), numeric(ncol(E))))
  V <- V / sqrt(rowSums(V^2))
  disp <- vapply(songs, function(i) mean(1 - E[stanzas$song == i, , drop = FALSE] %*% V[i, ]), numeric(1))
  rownames(V) <- songs
  list(V = V, dispersion = disp)
}

# sign convention: every PC oriented so it correlates positively with song number
# (makes runs comparable across backends; the sign itself carries no meaning)
layout_pca <- function(V, E, n_pc = 3) {
  pca <- prcomp(V, center = TRUE, scale. = FALSE)
  sgn <- sign(cor(pca$x[, 1:n_pc], seq_len(nrow(V))))
  sgn[sgn == 0] <- 1
  rot <- sweep(pca$rotation[, 1:n_pc], 2, sgn, "*")
  list(song = scale(V, center = pca$center, scale = FALSE) %*% rot
       , stanza = scale(E, center = pca$center, scale = FALSE) %*% rot
       , rotation = rot, center = pca$center
       , var_explained = (pca$sdev^2 / sum(pca$sdev^2))[1:n_pc])
}

layout_umap <- function(V, E, n_neighbors = 8, min_dist = 0.3, seeds = 1:5) {
  fits <- lapply(seeds, function(s) {
    set.seed(s)
    umap(V, n_neighbors = n_neighbors, min_dist = min_dist, metric = "cosine",
         init = "pca", n_threads = 1, ret_model = TRUE)
  })
  # stability: correlation of the 2-D pairwise distances across seeds (1 = identical maps)
  D <- lapply(fits, function(f) as.vector(dist(f$embedding)))
  stab <- combn(length(D), 2, function(ij) cor(D[[ij[1]]], D[[ij[2]]]))
  f <- fits[[1]]
  list(song = f$embedding, stanza = umap_transform(E, f), stability = stab)
}

# designed axes: axis = mean(+ pole anchors) - mean(- pole anchors); score = projection
# of the centred vectors. anchors_dt: axis, pole ("+"/"-"), text
layout_anchors <- function(V, E, anchors_dt, embed_fn) {
  A <- embed_fn(anchors_dt$text)
  axes <- unique(anchors_dt$axis)
  W <- vapply(axes, function(a) {
    w <- colMeans(A[anchors_dt$axis == a & anchors_dt$pole == "+", , drop = FALSE]) -
      colMeans(A[anchors_dt$axis == a & anchors_dt$pole == "-", , drop = FALSE])
    w / sqrt(sum(w^2))
  }, numeric(ncol(V)))
  ctr <- colMeans(V)
  list(song = scale(V, center = ctr, scale = FALSE) %*% W
       , stanza = scale(E, center = ctr, scale = FALSE) %*% W
       , axes = axes, cos_between_axes = sum(W[, 1] * W[, 2]))
}

path_length <- function(V, ord) sum(1 - rowSums(V[ord[-length(ord)], ] * V[ord[-1], ]))

test_order <- function(V, orders, n_perm = 20000) {
  set.seed(911)
  null <- replicate(n_perm, path_length(V, sample(nrow(V))))
  rbindlist(lapply(names(orders), function(nm) {
    L <- path_length(V, orders[[nm]])
    data.table(order = nm, path_length = L, null_mean = mean(null), null_sd = sd(null)
               , z = (L - mean(null)) / sd(null), p_shorter = mean(null <= L))
  }))
}

# the k stanzas at each end of an axis = what that axis "means"
axis_extremes <- function(stanzas, stanza_scores, axis_name, k = 6) {
  dt <- copy(stanzas)[, score := stanza_scores][order(score)]
  rbind(dt[1:k][, end := "-"], dt[(.N - k + 1):.N][, end := "+"])[
    , .(axis = axis_name, end, unit_id, title, score = round(score, 3), text)]
}

plot_trajectory <- function(song_dt, stanza_dt, xlab, ylab, title, subtitle, colour_lab, caption) {
  ggplot(song_dt, aes(x, y)) +
    geom_point(data = stanza_dt, colour = "grey60", size = 0.9, alpha = 0.6) +
    geom_link2(aes(colour = PC3, linewidth = dispersion, group = 1), lineend = "round", alpha = 0.9, n = 60) +
    geom_point(shape = 21, fill = "white", colour = "grey15", size = 5.2, stroke = 0.6) +
    geom_text(aes(label = song), size = 2.6, fontface = "bold") +
    geom_text_repel(aes(label = title), size = 3.1, box.padding = 0.55, point.padding = 0.6,
                    min.segment.length = 0.3, max.overlaps = Inf, seed = 1, colour = "grey20") +
    scale_colour_gradient2(low = "#2166AC", mid = "#E8E2D6", high = "#B2182B", midpoint = 0,
                           name = colour_lab) +
    scale_linewidth(range = c(1.2, 7), name = "within-song\ndispersion") +
    labs(x = xlab, y = ylab, title = title, subtitle = subtitle, caption = caption) +
    theme_minimal(base_size = 12) +
    theme(plot.title = element_text(face = "bold", size = 16),
          panel.grid.minor = element_blank(), legend.position = "right",
          plot.background = element_rect(fill = "white", colour = NA))
}
