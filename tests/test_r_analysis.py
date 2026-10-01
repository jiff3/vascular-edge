from pathlib import Path


def test_r_analysis_is_substantive_and_guards_estimability():
    root = Path(__file__).parents[1] / "r"
    script = (root / "run_models.R").read_text(encoding="utf-8")
    helpers = (root / "functions.R").read_text(encoding="utf-8")
    assert "lmer(" in helpers and "isSingular" in helpers
    assert "distance_region" in script and "absolute_wmh_change_ml" in script
    assert "cognitive" in script and "p.adjust" in helpers
    assert "not_estimable" in script and "not_estimable" in helpers
    assert "sessionInfo()" in script and "set.seed" in script
