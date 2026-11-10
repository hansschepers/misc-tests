setwd("C:/Users/Lenovo/Documents/misc-tests/muse/winterreise")
sink("work_step1_output.txt")
suppressPackageStartupMessages({
  library(data.table); library(glue); library(httr2); library(jsonlite)
  library(torch); library(tok); library(hfhub); library(safetensors)
})
for (f in list.files("R", full.names = TRUE)) source(f, encoding = "UTF-8")

wt <- fetch_winterreise_wikitext("cache/wikitext_raw.txt")
dir.create("cache", showWarnings = FALSE)
P <- parse_winterreise(wt)
print(P$stanzas[, .(n_stanza = .N, sung = sum(n_lines_sung), uniq = sum(n_lines_unique)), by = .(song, title, tempo)])
cat("\n", nrow(P$stanzas), "stanza units\n")
print(P$stanzas[song %in% c(1, 5, 24), .(unit_id, n_lines_sung, n_lines_unique, text)])

probe <- c("Ich bin so müde und will nur schlafen."
           , "I am so tired and just want to sleep."
           , "Erschöpft möchte ich mich hinlegen und ruhen."
           , "Die Post bringt heute einen Brief aus der Stadt."
           , "Der Lindenbaum am Brunnen vor dem Tore.")
E <- embed_texts(probe, backend = "rtorch", cache_dir = "cache/probe")
print(round(E %*% t(E), 3))
sink()
