##############################################################################
# load_score.R
# Load a MuseScore-extracted note table, enrich with derived columns,
# and provide helpers for analysis / round-trip back to .mscx
##############################################################################

library(data.table)

# ---- 1. Read and type the CSV ----------------------------------------------

#' Read a MuseScore note CSV into a data.table
#'
#' @param csv_path  path to the CSV produced by extract_notes.py
#' @return data.table with one row per note/rest event
read_score_csv <- function(csv_path) {

  dt <- fread(csv_path, na.strings = "")

  # --- coerce types ---------------------------------------------------------
  # staff_id        [1]         integer, staff index

  # part_name       [-]         character, e.g. "Sopraan 1"
  # measure         [1]         integer, bar number
  # voice           [1]         integer, voice within staff (1-based)
  # beat_pos        [beats]     numeric, onset within measure in quarter-notes
  # pitch_midi      [1]         integer 0-127, MIDI note number (NA for rests)
  # pitch_name      [-]         character, e.g. "Fs4" (NA for rests)
  # pitch_spelled   [-]         character, enharmonic spelling from tpc, e.g. "F#"
  # octave          [1]         integer, scientific octave (NA for rests)
  # duration_type   [-]         character, MuseScore name: whole/half/quarter/...
  # duration_beats  [beats]     numeric, duration in quarter-notes
  # dots            [1]         integer, number of augmentation dots
  # is_rest         [-]         logical
  # is_tied         [-]         logical, TRUE if note starts a tie to next

  # is_grace        [-]         logical, TRUE for grace/appoggiatura notes
  # ts_num          [1]         integer, time-sig numerator at this event
  # ts_den          [1]         integer, time-sig denominator at this event

  int_cols <- c("staff_id", "measure", "voice", "pitch_midi", "octave",
                "dots", "ts_num", "ts_den")
  for (col in int_cols) {
    if (col %in% names(dt)) set(dt, j = col, value = as.integer(dt[[col]]))
  }

  num_cols <- c("beat_pos", "duration_beats")
  for (col in num_cols) {
    if (col %in% names(dt)) set(dt, j = col, value = as.numeric(dt[[col]]))
  }

  bool_cols <- c("is_rest", "is_tied", "is_grace")
  for (col in bool_cols) {
    if (col %in% names(dt)) {
      set(dt, j = col, value = dt[[col]] %in% c("True", "TRUE", TRUE))
    }
  }

  dt
}


# ---- 2. Derived columns ---------------------------------------------------

#' Enrich a score data.table with derived analysis columns
#'
#' @param dt  data.table from read_score_csv()
#' @return dt (modified in place) with extra columns
enrich_score <- function(dt) {

  # --- absolute time in quarter-note beats from start of piece [beats] ------
  # first, cumulative beat offset per measure (same for all staves)
  meas_dur <- dt[, .(measure_beats = ts_num[1] * (4.0 / ts_den[1])),
                 by = measure]
  meas_dur[, cum_beat := cumsum(shift(measure_beats, 1L, fill = 0))]
  # cum_beat = beat offset at START of each measure [beats]

  dt <- merge(dt, meas_dur[, .(measure, cum_beat)], by = "measure", sort = FALSE)
  dt[, time_beat := cum_beat + beat_pos]        # [beats] absolute onset
  dt[, time_s    := time_beat * (60.0 / 66.0)]  # [s] at quarter=66 BPM

  # --- interval from previous note in same voice [semitones] ----------------
  setorder(dt, staff_id, voice, time_beat)
  dt[is_rest == FALSE,
     interval := pitch_midi - shift(pitch_midi, 1L),
     by = .(staff_id, voice)]
  # interval [semitones], positive = ascending, NA for first note

  # --- pitch class (0=C, ..., 11=B) [1] ------------------------------------
  dt[is_rest == FALSE, pitch_class := pitch_midi %% 12L]

  # --- duration in seconds [s] -----------------------------------------------
  dt[, duration_s := duration_beats * (60.0 / 66.0)]

  setorder(dt, staff_id, voice, measure, beat_pos)
  dt
}


# ---- 3. Round-trip: data.table -> .mscx XML --------------------------------

#' Convert a note data.table back to a minimal MuseScore 4 .mscx XML string
#'
#' This produces a valid but minimal score — no layout, no lyrics, no dynamics.
#' Suitable for importing into MuseScore for further editing.
#'
#' @param dt          data.table with columns: staff_id, part_name, measure,
#'                    voice, beat_pos, pitch_midi, pitch_spelled, octave,
#'                    duration_type, duration_beats, dots, is_rest, is_tied,
#'                    ts_num, ts_den
#' @param title       character, score title
#' @param composer    character
#' @param tempo_bpm   numeric, quarter-note BPM [beats/min]
#' @param key_fifths  integer, key signature in fifths (e.g. 4 = E major)
#' @return character, a complete .mscx XML document
dt_to_mscx <- function(dt,
                        title      = "Untitled",
                        composer   = "",
                        tempo_bpm  = 66,
                        key_fifths = 0L) {

  # --- helpers for XML building (no dependency on xml2) ---------------------
  indent <- function(level) paste(rep("  ", level), collapse = "")

  xml_tag <- function(tag, text, level = 0) {
    paste0(indent(level), "<", tag, ">", text, "</", tag, ">")
  }

  # --- pitch_midi -> tpc (tonal pitch class for MuseScore) ------------------
  # tpc encodes the note on the line of fifths: C=14, G=15, D=16, ...
  # For simplicity: map pitch class to the most common tpc in the key
  midi_to_tpc <- function(midi, key = key_fifths) {
    # base tpc for each pitch class (no accidentals = C major)
    base <- c(14L, 21L, 16L, 23L, 18L, 13L, 20L, 15L, 22L, 17L, 24L, 19L)
    #         C    C#   D    D#   E    F    F#   G    G#   A    A#   B
    pc <- (as.integer(midi) %% 12L) + 1L
    base[pc]
  }

  # --- duration_type -> MuseScore durationType string -----------------------
  # already stored as "whole", "half", "quarter", "eighth", etc. — pass through

  # --- build XML lines ------------------------------------------------------
  lines <- character()
  add   <- function(...) lines <<- c(lines, paste0(...))

  add('<?xml version="1.0" encoding="UTF-8"?>')
  add('<museScore version="4.70">')
  add('  <Score>')
  add('    <Division>480</Division>')  # ticks per quarter [ticks/beat]
  add(xml_tag("metaTag", title,    3), ' <!-- name="workTitle" -->')
  add(xml_tag("metaTag", composer, 3), ' <!-- name="composer" -->')

  # --- Parts ----------------------------------------------------------------
  staff_ids  <- sort(unique(dt$staff_id))
  part_names <- dt[, .(part_name = part_name[1]), by = staff_id]
  setorder(part_names, staff_id)

  # instrument guessing from part name
  guess_instrument <- function(pname) {
    pname_lc <- tolower(pname)
    if (grepl("sop", pname_lc))   return(list(id = "soprano", clef = "G",    midi_prog = 52L))
    if (grepl("alt",  pname_lc))  return(list(id = "alto",    clef = "G",    midi_prog = 52L))
    if (grepl("ten",  pname_lc))  return(list(id = "tenor",   clef = "G8vb", midi_prog = 52L))
    if (grepl("bas|bass", pname_lc)) return(list(id = "bass", clef = "F",    midi_prog = 52L))
    list(id = "voice.soprano", clef = "G", midi_prog = 52L)
  }

  for (i in seq_len(nrow(part_names))) {
    sid   <- part_names$staff_id[i]
    pname <- part_names$part_name[i]
    inst  <- guess_instrument(pname)

    add('    <Part id="', sid, '">')
    add('      <Staff>')
    add('        <StaffType group="pitched"><name>stdNormal</name></StaffType>')
    if (inst$clef != "G") {
      add('        <defaultClef>', inst$clef, '</defaultClef>')
    }
    add('      </Staff>')
    add('      <trackName>', pname, '</trackName>')
    add('      <Instrument id="', inst$id, '">')
    add('        <longName>', pname, '</longName>')
    add('        <instrumentId>voice.', inst$id, '</instrumentId>')
    if (inst$clef != "G") add('        <clef>', inst$clef, '</clef>')
    add('        <Channel><program value="', inst$midi_prog, '"/></Channel>')
    add('      </Instrument>')
    add('    </Part>')
  }

  # --- Staves with measures --------------------------------------------------
  for (sid in staff_ids) {
    add('    <Staff id="', sid, '">')

    # title frame on first staff only
    if (sid == staff_ids[1]) {
      add('      <VBox><height>10</height>')
      add('        <Text><style>title</style><text>', title, '</text></Text>')
      if (nchar(composer) > 0) {
        add('        <Text><style>composer</style><text>', composer, '</text></Text>')
      }
      add('      </VBox>')
    }

    staff_dt <- dt[staff_id == sid]
    measures <- sort(unique(staff_dt$measure))

    for (mi in measures) {
      meas_dt <- staff_dt[measure == mi]
      add('      <Measure>')

      # voices in this measure
      voices <- sort(unique(meas_dt$voice))
      for (vi in voices) {
        add('        <voice>')

        voice_dt <- meas_dt[voice == vi]
        setorder(voice_dt, beat_pos)

        # key sig + time sig + tempo on first measure, first staff, first voice
        if (mi == measures[1] && vi == 1L) {
          add('          <KeySig><concertKey>', key_fifths, '</concertKey></KeySig>')
          ts_n <- voice_dt$ts_num[1]
          ts_d <- voice_dt$ts_den[1]
          add('          <TimeSig><sigN>', ts_n, '</sigN><sigD>', ts_d, '</sigD></TimeSig>')
          if (sid == staff_ids[1]) {
            add('          <Tempo><tempo>', tempo_bpm / 60.0, '</tempo></Tempo>')
          }
        }

        # check for time-sig change at measure start vs previous
        if (mi > measures[1]) {
          prev_meas <- staff_dt[measure == mi - 1L]
          if (nrow(prev_meas) > 0) {
            prev_ts <- paste0(prev_meas$ts_num[1], "/", prev_meas$ts_den[1])
            curr_ts <- paste0(voice_dt$ts_num[1], "/", voice_dt$ts_den[1])
            if (prev_ts != curr_ts && vi == 1L) {
              add('          <TimeSig><sigN>', voice_dt$ts_num[1],
                  '</sigN><sigD>', voice_dt$ts_den[1], '</sigD></TimeSig>')
            }
          }
        }

        # notes and rests
        for (ri in seq_len(nrow(voice_dt))) {
          row <- voice_dt[ri]

          if (row$is_rest) {
            add('          <Rest>')
            add('            <durationType>', row$duration_type, '</durationType>')
            if (row$duration_type == "measure") {
              add('            <duration>', row$ts_num, '/', row$ts_den, '</duration>')
            }
            add('          </Rest>')
          } else {
            add('          <Chord>')
            add('            <durationType>', row$duration_type, '</durationType>')
            if (!is.na(row$dots) && row$dots > 0L) {
              for (d in seq_len(row$dots)) add('            <dot/>')
            }
            if (!is.na(row$is_tied) && row$is_tied) {
              add('            <Spanner type="Tie"><next><location/></next></Spanner>')
            }
            add('            <Note>')
            add('              <pitch>', row$pitch_midi, '</pitch>')
            tpc_val <- midi_to_tpc(row$pitch_midi)
            add('              <tpc>', tpc_val, '</tpc>')
            add('            </Note>')
            add('          </Chord>')
          }
        }

        add('        </voice>')
      }
      add('      </Measure>')
    }
    add('    </Staff>')
  }

  add('  </Score>')
  add('</museScore>')

  paste(lines, collapse = "\n")
}


#' Write a data.table score to a .mscz file (ZIP containing .mscx)
#'
#' @param dt          data.table from read_score_csv / enrich_score
#' @param output_path path for the .mscz output file
#' @param title       score title
#' @param composer    composer name
#' @param tempo_bpm   quarter-note BPM [beats/min]
#' @param key_fifths  key signature in fifths (4 = E major, 0 = C major, -1 = F major, ...)
write_mscz <- function(dt,
                        output_path,
                        title      = "Untitled",
                        composer   = "",
                        tempo_bpm  = 66,
                        key_fifths = 0L) {

  mscx_xml <- dt_to_mscx(dt,
                           title      = title,
                           composer   = composer,
                           tempo_bpm  = tempo_bpm,
                           key_fifths = key_fifths)

  # MuseScore 4 .mscz = ZIP with the .mscx inside
  base_name <- tools::file_path_sans_ext(basename(output_path))
  mscx_name <- paste0(base_name, ".mscx")

  tmp_dir <- tempdir()
  mscx_path <- file.path(tmp_dir, mscx_name)
  writeLines(mscx_xml, mscx_path)

  # also write the mandatory META-INF/container.xml
  meta_dir <- file.path(tmp_dir, "META-INF")
  dir.create(meta_dir, showWarnings = FALSE)
  container_xml <- paste0(
    '<?xml version="1.0" encoding="UTF-8"?>\n',
    '<container><rootfiles>',
    '<rootfile full-path="', mscx_name, '"/>',
    '</rootfiles></container>'
  )
  writeLines(container_xml, file.path(meta_dir, "container.xml"))

  # create ZIP
  old_wd <- setwd(tmp_dir)
  on.exit(setwd(old_wd))
  zip(output_path, files = c(mscx_name, file.path("META-INF", "container.xml")))
  setwd(old_wd)

  message("Wrote ", output_path)
  invisible(output_path)
}


# ---- 4. Example usage (run interactively) ----------------------------------
if (FALSE) {

  # --- load and enrich ------------------------------------------------------
  notes <- read_score_csv("muse/o_magnum_mysterium_notes.csv")
  notes <- enrich_score(notes)

  # --- basic summaries ------------------------------------------------------
  notes[is_rest == FALSE, .N, by = part_name]

  # pitch range per part [semitones]
  notes[is_rest == FALSE,
        .(lo = min(pitch_midi), hi = max(pitch_midi), range = max(pitch_midi) - min(pitch_midi)),
        by = part_name]

  # interval histogram [semitones]
  notes[is_rest == FALSE & !is.na(interval),
        .N,
        by = .(part_name, interval)][order(part_name, interval)]

  # --- piano-roll plot ------------------------------------------------------
  library(ggplot2)
  pList <- list()

  p1 <- ggplot(notes[is_rest == FALSE],
               aes(x    = time_beat,                        # [beats]
                   y    = pitch_midi,                       # [MIDI]
                   xend = time_beat + duration_beats,
                   yend = pitch_midi,
                   color = part_name)) +
    geom_segment(linewidth = 2) +
    labs(x = "Time [quarter-note beats]", y = "Pitch [MIDI note number]",
         color = "Part", title = "O Magnum Mysterium — piano roll") +
    theme_minimal()
  pList$piano_roll <- p1

  # --- round-trip test: write back to .mscz --------------------------------
  write_mscz(notes,
              output_path = "muse/o_magnum_roundtrip.mscz",
              title       = "O Magnum Mysterium (round-trip)",
              composer    = "Michael John Trotta",
              tempo_bpm   = 66,
              key_fifths  = 4L)
}
