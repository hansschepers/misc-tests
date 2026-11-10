# Song-cycle texts from de.wikisource (public domain), parsed into song x stanza units.
# A cycle = a loader returning list(lines = data.table(song, title, tempo, stanza, line),
# alt_order = <the poet's own order, in composer numbers>, alt_order_name).
# Configs: cycle_config(). New cycle -> write load_<cycle>() + add a config entry.

wiki_wikitext <- function(page, cache_file, host = "de.wikisource.org") {
  if (file.exists(cache_file)) {
    return(paste(readLines(cache_file, encoding = "UTF-8", warn = FALSE), collapse = "\n"))
  }
  b <- request(glue("https://{host}/w/api.php")) |>
    req_url_query(action = "parse", page = page, prop = "wikitext", format = "json", redirects = 1) |>
    req_user_agent("song-cycle-semantic-path/0.1 (R httr2)") |>
    req_throttle(capacity = 1, fill_time_s = 1) |>   # wikimedia rate limit: ~1 request/s
    req_retry(max_tries = 5) |>                       # honours Retry-After on 429
    req_perform() |>
    resp_body_json()
  if (!is.null(b$error)) stop(glue("wikisource '{page}': {b$error$info}"))
  wt <- b$parse$wikitext[[1]]
  dir.create(dirname(cache_file), showWarnings = FALSE, recursive = TRUE)
  con <- file(cache_file, open = "w", encoding = "UTF-8")
  writeLines(wt, con)
  close(con)
  wt
}

extract_poem <- function(wikitext) sub("(?s).*?<poem>(.*?)</poem>.*", "\\1", wikitext, perl = TRUE)

# one <poem> body -> list(tempo, lines = data.table(stanza, line)). Blank line = stanza break;
# a line wholly in ''italics'' = performance marking (Schubert/Friedlaender), kept apart.
poem_lines <- function(poem) {
  poem <- gsub("\\{\\{[^}]*\\}\\}", "", poem)   # {{idt}}, {{Zeile|5}}, {{Seite|4||x.png}} ...
  poem <- gsub("<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", poem, perl = TRUE)
  ln <- strsplit(poem, "\n")[[1]]
  ln <- trimws(gsub("[\\h]+", " ", ln, perl = TRUE))
  ln <- ln[!grepl("^'''[^']+'''$", ln)]          # bold title line (Hatfield edition)
  is_tempo <- grepl("^''.*''$", ln)
  tempo <- if (any(is_tempo)) gsub("''", "", ln[is_tempo][1]) else NA_character_
  ln <- ln[!is_tempo]
  ln <- gsub("'{2,}", "", ln)                   # inline emphasis ('''sie''' = gesperrt) -> plain
  st <- cumsum(ln == "" & c(TRUE, head(ln, -1) != ""))
  keep <- ln != ""
  bad <- ln[keep][grepl("\\{\\{|\\}\\}|\\[\\[|<[a-z/]|^[0-9 ]+$|''", ln[keep])]
  if (length(bad)) stop(glue("poem_lines: markup/number left in text: {paste(head(bad, 3), collapse = ' | ')}"))
  list(tempo = tempo, lines = data.table(stanza = match(st[keep], unique(st[keep])), line = ln[keep]))
}

# lines -> stanza units. Dedup: composers repeat lines/stanzas; the unit is the POET's stanza
# (each distinct line once, first-occurrence order); n_lines_sung keeps the repetition count.
lines_to_stanzas <- function(lines_dt) {
  lines_dt <- copy(lines_dt)
  lines_dt[, key := trimws(tolower(gsub("[[:punct:]–’ ]+", " ", line)))]
  st <- lines_dt[, .(text = paste(line[!duplicated(key)], collapse = " ")
                     , first_line = line[1]
                     , n_lines_sung = .N
                     , n_lines_unique = sum(!duplicated(key))
                     , skey = paste(unique(key), collapse = "|")
                     ), by = .(song, title, tempo, stanza)]
  # drop a stanza whose distinct lines all occurred earlier in the same song
  st[, dup := {
    seen <- character(0)
    out <- logical(.N)
    for (j in seq_len(.N)) {
      k <- strsplit(skey[j], "|", fixed = TRUE)[[1]]
      out[j] <- all(k %in% seen)
      seen <- c(seen, k)
    }
    out
  }, by = song]
  st <- st[dup == FALSE][, c("dup", "skey") := NULL]
  st[, stanza := seq_len(.N), by = song]
  st[, unit_id := paste0(song, ".", stanza)]  # glue() cannot see data.table columns in j
  st[]
}

# ---- Winterreise: Schubert/Friedlaender edition, one page, 24 '== n. Title ==' sections --
load_winterreise <- function(cache_dir) {
  wt <- wiki_wikitext("Winterreise_(Schubert/Friedlaender)", file.path(cache_dir, "wikitext_raw.txt"))
  head_re <- "(?m)^== *([0-9]+)\\. *(.+?) *== *$"
  heads <- regmatches(wt, gregexpr(head_re, wt, perl = TRUE))[[1]]
  bodies <- strsplit(wt, head_re, perl = TRUE)[[1]][-1]
  stopifnot(length(heads) == 24, length(bodies) == 24)
  lines <- rbindlist(lapply(seq_along(heads), function(i) {
    pl <- poem_lines(extract_poem(bodies[i]))
    data.table(song = i, title = sub("^== *[0-9]+\\. *(.+?) *== *$", "\\1", heads[i], perl = TRUE),
               tempo = pl$tempo, pl$lines)
  }))
  # Müller's final 1824 order in Schubert numbers (from memory -- VERIFY)
  list(lines = lines, alt_order_name = "mueller_1824",
       alt_order = c(1, 2, 3, 4, 5, 13, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 9, 10, 23, 11, 12, 22, 24))
}

# ---- Dichterliebe: Heine, Buch der Lieder (1827), one wikisource page per poem -----------
dichterliebe_songs <- function() {
  data.table(
    song = 1:16
    , title = c("Im wunderschönen Monat Mai", "Aus meinen Tränen sprießen",
                "Die Rose, die Lilie, die Taube", "Wenn ich in deine Augen seh",
                "Ich will meine Seele tauchen", "Im Rhein, im heiligen Strome", "Ich grolle nicht",
                "Und wüßten's die Blumen", "Das ist ein Flöten und Geigen",
                "Hör ich das Liedchen klingen", "Ein Jüngling liebt ein Mädchen",
                "Am leuchtenden Sommermorgen", "Ich hab im Traum geweinet",
                "Allnächtlich im Traume", "Aus alten Märchen", "Die alten, bösen Lieder")
    , page = c("Im wunderschönen Monat Mai", "Aus meinen Thränen sprießen",
               "Die Rose, die Lilie, die Taube, die Sonne", "Wenn ich in deine Augen seh’",
               "Ich will meine Seele tauchen", "Im Rhein, im heiligen Strome",
               "Ich grolle nicht, und wenn das Herz auch bricht", "Und wüßten’s die Blumen, die kleinen",
               "Das ist ein Flöten und Geigen", "Hör’ ich das Liedchen klingen",
               "Ein Jüngling liebt ein Mädchen", "Am leuchtenden Sommermorgen",
               "Ich hab’ im Traum’ geweinet", "Allnächtlich im Traume seh’ ich dich",
               "Aus alten Mährchen winkt es", "Die alten, bösen Lieder"))
}

load_dichterliebe <- function(cache_dir) {
  S <- dichterliebe_songs()
  lines <- rbindlist(lapply(S$song, function(i) {
    wt <- wiki_wikitext(S$page[i], file.path(cache_dir, "dichterliebe", glue("{sprintf('%02d', i)}.txt")))
    pl <- poem_lines(extract_poem(wt))
    data.table(song = i, title = S$title[i], tempo = pl$tempo, pl$lines)
  }))
  # Heine's own order: position in the Lyrisches Intermezzo list of 'Buch der Lieder'
  bdl <- wiki_wikitext("Buch der Lieder", file.path(cache_dir, "buch_der_lieder.txt"))
  sec <- sub("(?s).*== *Lyrisches Intermezzo[^\n]*\n(.*?)\n== .*", "\\1", bdl, perl = TRUE)
  li <- regmatches(sec, gregexpr("(?m)^#\\[\\[([^]|]+)", sec, perl = TRUE))[[1]]
  li <- sub("^#\\[\\[", "", li)
  heine_no <- match(S$page, li)
  stopifnot(!anyNA(heine_no))
  cat(glue("Dichterliebe -> Heine Lyrisches Intermezzo numbers: {paste(heine_no, collapse = ' ')}"), "\n")
  list(lines = lines, alt_order_name = "heine_intermezzo", alt_order = S$song[order(heine_no)],
       heine_no = heine_no)
}

# ---- Die schöne Müllerin: Müller (Hatfield 1906 critical ed.), one page per poem ---------
# Schubert set 20 of Müller's 25, in Müller's order (omitted: Prolog, Das Mühlenleben,
# Erster Schmerz letzter Scherz, Blümlein Vergißmein, Epilog) -> no alternative order to test.
muellerin_songs <- function() {
  data.table(
    song = 1:20
    , title = c("Das Wandern", "Wohin?", "Halt!", "Danksagung an den Bach", "Am Feierabend",
                "Der Neugierige", "Ungeduld", "Morgengruß", "Des Müllers Blumen", "Tränenregen",
                "Mein!", "Pause", "Mit dem grünen Lautenbande", "Der Jäger", "Eifersucht und Stolz",
                "Die liebe Farbe", "Die böse Farbe", "Trockne Blumen", "Der Müller und der Bach",
                "Des Baches Wiegenlied")
    , page = c("Wanderschaft", "Wohin", "Halt!", "Danksagung an den Bach", "Am Feierabend",
               "Der Neugierige", "Ungeduld", "Morgengruß", "Des Müllers Blumen", "Thränenregen",
               "Mein!", "Pause", "Mit dem grünen Lautenbande", "Der Jäger (Müller)", "Eifersucht und Stolz",
               "Die liebe Farbe", "Die böse Farbe", "Trockne Blumen", "Der Müller und der Bach",
               "Des Baches Wiegenlied"))
}

load_muellerin <- function(cache_dir) {
  S <- muellerin_songs()
  lines <- rbindlist(lapply(S$song, function(i) {
    wt <- wiki_wikitext(S$page[i], file.path(cache_dir, "muellerin", glue("{sprintf('%02d', i)}.txt")))
    pl <- poem_lines(extract_poem(wt))
    data.table(song = i, title = S$title[i], tempo = pl$tempo, pl$lines)
  }))
  list(lines = lines, alt_order_name = NULL, alt_order = NULL)
}

cycle_config <- function(cycle) {
  switch(cycle
    , winterreise = list(
        load = load_winterreise, label = "Winterreise (Müller / Schubert 1827)"
        , composer_order = "Schubert's order"
        , title_en = c("Good Night", "The Weathervane", "Frozen Tears", "Numbness",
            "The Linden Tree", "Flood", "On the River", "Looking Back", "Will-o'-the-Wisp", "Rest",
            "Dream of Spring", "Loneliness", "The Post", "The Grey Head", "The Crow", "Last Hope",
            "In the Village", "The Stormy Morning", "Delusion", "The Signpost", "The Inn", "Courage!",
            "The Mock Suns", "The Hurdy-Gurdy Man"))
    , dichterliebe = list(
        load = load_dichterliebe, label = "Dichterliebe (Heine / Schumann 1840)"
        , composer_order = "Schumann's order"
        , title_en = c("In the wondrous month of May", "From my tears spring", "The rose, the lily, the dove",
            "When I look into your eyes", "I will dip my soul", "In the Rhine, the holy river",
            "I bear no grudge", "And if the little flowers knew", "There is a fluting and fiddling",
            "When I hear the little song", "A youth loves a maiden", "On a bright summer morning",
            "I wept in my dream", "Nightly in my dreams", "From old fairy tales", "The old, evil songs"))
    , muellerin = list(
        load = load_muellerin, label = "Die schöne Müllerin (Müller / Schubert 1823)"
        , composer_order = "Schubert's order (= Müller's, 5 poems omitted)"
        , title_en = c("Wandering", "Whither?", "Halt!", "Thanks to the Brook", "After Work",
            "The Inquisitive One", "Impatience", "Morning Greeting", "The Miller's Flowers", "Rain of Tears",
            "Mine!", "Pause", "With the Green Lute Ribbon", "The Hunter", "Jealousy and Pride",
            "The Beloved Colour", "The Hateful Colour", "Withered Flowers", "The Miller and the Brook",
            "The Brook's Lullaby"))
    , stop(glue("unknown cycle '{cycle}'")))
}
