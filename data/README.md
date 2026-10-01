# Data directories (not tracked in Git)

`data/` is the local boundary for raw, non-versioned research data. It is
excluded by `.gitignore`; never add participant images, TSVs, or DataLad annex
objects to this repository.

Suggested layout:

```text
data/
  ds004856/                 # OpenNeuro/DataLad metadata clone + selected annex objects
derivatives/
  vascular_edge/            # generated subject-level derivatives, also ignored
results/
  cohort_manifest.{json,csv} # non-identifying selection plan; may be retained locally
```

Discovery reads OpenNeuro's metadata tree and transfers no NIfTI data. Selection
writes a manifest first. Downloading is explicitly selective: it asks DataLad
for only files named in included manifest rows. OpenNeuro's official CLI creates
the local DataLad dataset; configure its git-annex/OpenNeuro remote according to
the OpenNeuro documentation before using a non-dry-run download.

Do not place manually de-identified data from another source in this directory
without documenting provenance and access conditions separately.
