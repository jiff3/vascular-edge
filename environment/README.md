# Environment

The supported baseline is Python 3.11 on Windows, macOS, or Linux. Create an
isolated environment and install editable development dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The core dependency set is intentionally CPU-first. No deep-learning framework,
GPU runtime, or full OpenNeuro clone is required. For reproducibility, record
the output of `python --version` and `pip freeze` with every analysis run.

Optional external tools may be configured later for validated production-grade
brain extraction/registration or a pretrained WMH model. The built-in fallback
is designed for transparent small-scale exploratory processing, not clinical use.
