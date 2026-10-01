# Data access and selective downloading

This project targets the Dallas Lifespan Brain Study OpenNeuro record
`ds004856` where it is available. Dataset content and participant permissions
can change; consult the dataset landing page, README, and Data Use Agreement
before downloading or redistributing any data.

Use `vascular-edge discover --bids-root <local-bids-directory>` after obtaining
a small BIDS subset. The command inventories available files instead of assuming
that every participant has T1w, FLAIR, ASL, or hypercapnia/BOLD imaging.

For controlled downloads, write a manifest of explicit HTTPS archive URLs:

```yaml
download:
  files:
    - url: https://example.org/sub-001_T1w.nii.gz
      path: sub-001/anat/sub-001_T1w.nii.gz
```

Then run `vascular-edge download --manifest configs/download.yaml --output data/ds004856`.
The downloader rejects destination paths outside the requested output directory.
