# OpenNeuro / DLBS selective workflow

The remote discovery implementation uses OpenNeuro's documented GraphQL snapshot
file-tree API. It lists metadata only; it does not fetch annexed image contents.
DLBS is longitudinal, but protocol availability must be inferred from the live
tree for each subject/session. Dataset version, filenames, and availability are
therefore recorded in a generated manifest rather than hard-coded.

```powershell
# Remote metadata inventory — no imaging download
vascular-edge discover --remote --config configs/cohort.example.yaml --output results/remote_inventory.json
vascular-edge select-cohort --inventory results/remote_inventory.json --config configs/cohort.example.yaml
vascular-edge download --manifest results/cohort_manifest.json --bids-root data/ds004856 --dry-run
```

For actual transfer, install Deno, OpenNeuro CLI, DataLad, git-annex, and the
OpenNeuro special remote. The official OpenNeuro CLI can create a metadata/DataLad
dataset (`openneuro download ds004856 data/ds004856`); then this project invokes
`datalad get` only on paths in the selected manifest. This is deliberately a
separate, user-controlled prerequisite because it may require OpenNeuro login and
local special-remote configuration. Validate after transfer:

```powershell
vascular-edge validate-data --manifest results/cohort_manifest.json --bids-root data/ds004856
```
