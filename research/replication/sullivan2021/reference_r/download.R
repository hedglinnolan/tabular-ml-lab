# Sullivan 2021 reference (R only): download the public NHANES files and write the manifest.
#
# Run from the repository root:
#   Rscript research/replication/sullivan2021/reference_r/download.R
#
# Each data file (.xpt) and its codebook page (.htm) is fetched from wwwn.cdc.gov into
# research/data/nhanes/<cycle>/ (ignored by git). The committed manifest records, per file, the
# URL, the download date, the size, the SHA-256, the dates the CDC codebook page states for the
# file ("First Published", "Last Revised") and the server's Last-Modified header. A file already on
# disk is not fetched again; its first download date is kept from the manifest.

suppressPackageStartupMessages({
  library(digest)
  library(jsonlite)
})

options(timeout = 600)

data_root <- file.path("research", "data", "nhanes")
manifest_path <- file.path("research", "replication", "sullivan2021", "reference_r", "manifest.json")

cycles <- list(
  list(cycle = "2011-2012", year = 2011, suffix = "G"),
  list(cycle = "2013-2014", year = 2013, suffix = "H"),
  list(cycle = "2015-2016", year = 2015, suffix = "I"),
  list(cycle = "2017-2018", year = 2017, suffix = "J")
)

# Insulin is in GLU_G for 2011-2012 (INS_G does not exist); from 2013-2014 it has its own file.
components <- function(suffix) {
  base <- c("DEMO", "DBQ", "DR1TOT", "DR2TOT", "GLU", "GHB", "BMX", "SMQ", "PAQ", "DIQ")
  if (suffix != "G") base <- c(base, "INS")
  paste0(base, "_", suffix)
}

old <- if (file.exists(manifest_path)) fromJSON(manifest_path, simplifyVector = FALSE)$files else list()
old_by_path <- setNames(old, vapply(old, function(f) f$path, ""))

page_dates <- function(htm_path) {
  txt <- paste(readLines(htm_path, warn = FALSE), collapse = " ")
  grab <- function(label) {
    m <- regmatches(txt, regexpr(paste0(label, ":\\s*[^<]+"), txt))
    if (length(m) == 0) NA_character_ else trimws(sub(paste0(label, ":\\s*"), "", m))
  }
  list(first_published = grab("First Published"), last_revised = grab("Last Revised"))
}

last_modified <- function(url) {
  h <- tryCatch(curlGetHeaders(url), error = function(e) character())
  lm <- grep("^last-modified:", h, ignore.case = TRUE, value = TRUE)
  if (length(lm) == 0) NA_character_ else trimws(sub("^[^:]+:", "", lm[length(lm)]))
}

entries <- list()
for (cy in cycles) {
  dir.create(file.path(data_root, cy$cycle), recursive = TRUE, showWarnings = FALSE)
  for (name in components(cy$suffix)) {
    for (ext in c("htm", "xpt")) {
      url <- sprintf("https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/%d/DataFiles/%s.%s", cy$year, name, ext)
      path <- file.path(data_root, cy$cycle, paste0(name, ".", ext))
      prior <- old_by_path[[path]]
      if (!file.exists(path)) {
        status <- download.file(url, path, mode = "wb", quiet = TRUE, method = "libcurl")
        if (status != 0) stop("download failed: ", url)
        downloaded <- format(Sys.time(), "%Y-%m-%d", tz = "UTC")
      } else {
        downloaded <- if (!is.null(prior)) prior$downloaded else format(file.mtime(path), "%Y-%m-%d", tz = "UTC")
      }
      # A missing file answers with an HTML error page, not a 404 status: refuse it.
      head_bytes <- readBin(path, "raw", 80)
      if (ext == "xpt" && !startsWith(rawToChar(head_bytes[1:40]), "HEADER RECORD*******LIBRARY HEADER")) {
        stop("not a SAS transport file: ", url)
      }
      entries[[length(entries) + 1]] <- list(
        cycle = cy$cycle,
        file = paste0(name, ".", ext),
        kind = if (ext == "xpt") "data (SAS transport)" else "codebook page",
        url = url,
        path = path,
        downloaded = downloaded,
        size_bytes = unname(file.size(path)),
        sha256 = digest(file = path, algo = "sha256"),
        http_last_modified = if (!is.null(prior) && !is.null(prior$http_last_modified)) prior$http_last_modified else last_modified(url)
      )
      cat(sprintf("%-10s %-14s %10d\n", cy$cycle, basename(path), file.size(path)))
    }
  }
}

# The codebook page's stated dates belong to both the page and its data file.
for (i in seq_along(entries)) {
  htm <- sub("\\.xpt$", ".htm", entries[[i]]$path)
  d <- page_dates(htm)
  entries[[i]]$cdc_first_published <- d$first_published
  entries[[i]]$cdc_last_revised <- d$last_revised
}

manifest <- list(
  description = paste(
    "Public NHANES files downloaded for the blind R reference of Sullivan et al. 2021",
    "(Am J Prev Med 61:563). The files live under research/data/ and are not committed."
  ),
  source = "wwwn.cdc.gov, no identifier in any request",
  cdc_dates = "cdc_first_published and cdc_last_revised are the 'First Published' and 'Last Revised' lines of the file's codebook page",
  files = entries
)
writeLines(toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, na = "null"), manifest_path)
cat(sprintf("%d files, %.1f MB\n", length(entries), sum(vapply(entries, function(e) e$size_bytes, 0)) / 1e6))
