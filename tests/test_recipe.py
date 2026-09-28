import json

import pytest

from mfa import recipe as R


def test_default_recipe_is_valid_and_landmark_only():
    r = R.default_recipe()
    assert set(r) == set(R.RECIPE_FIELDS)
    assert r["temporal_src"] in R.TEMPORAL_SRC_OPTIONS
    assert r["temporal_dst"] in R.TEMPORAL_DST_OPTIONS
    assert r["temporal_corridor"] in R.TEMPORAL_CORRIDORS
    assert r["temporal_extend"] in R.TEMPORAL_EXTEND_OPTIONS
    assert r["calcarine_extend"] in R.CALCARINE_EXTEND_OPTIONS
    assert R.recipe_str(r).startswith("slit_width=")
    assert R.recipe_code(r).startswith(f"w{r['slit_width']}_")


def test_parse_overrides_and_file(tmp_path):
    r = R.parse_recipe("temporal_corridor=medial,slit_width=3")
    assert r["temporal_corridor"] == "medial" and r["slit_width"] == 3
    fn = tmp_path / "r.json"
    fn.write_text(json.dumps({"recipe": r}))
    r2 = R.parse_recipe(f"{fn},temporal_extend=ITC_far_TG")
    assert r2["temporal_corridor"] == "medial" and r2["temporal_extend"] == "ITC_far_TG"
    assert R.recipe_key(r2) != R.recipe_key(r)
    assert R.load_recipe_file(str(fn)) == r


@pytest.mark.parametrize("bad", [
    "slit_width=0",
    "temporal_corridor=diagonal",
    "temporal_src=MTL_Amy_centroid",   # must start with nearest_
    "temporal_dst=TG_centroid",        # dst must be a geodesic anchor
    "temporal_src=nearest_TG_ant",     # coordinate extremes are not anatomical landmarks
    "temporal_dst=TG_ant",
    "frontal_dst=OFC_ant",
    "temporal_extend=TG_post",
    "temporal_extend=ITC_frac_0.5",    # fraction along a coordinate axis
    "temporal_extend=mm:7",            # millimetre extensions are not landmark rules
    "calcarine_extend=mm:3",
    "unknown_field=1",
])
def test_invalid_recipes(bad):
    with pytest.raises(ValueError):
        R.parse_recipe(bad)


def test_missing_field():
    with pytest.raises(ValueError):
        R.validate_recipe({"slit_width": 1})


def test_parse_slit_width():
    assert R.parse_slit_width(None) == {"lh": None, "rh": None}
    assert R.parse_slit_width("2") == {"lh": 2, "rh": 2}
    assert R.parse_slit_width("auto") == {"lh": "auto", "rh": "auto"}
    assert R.parse_slit_width("lh=1,rh=auto") == {"lh": 1, "rh": "auto"}
    assert R.parse_slit_width("rh=3") == {"lh": None, "rh": 3}
    with pytest.raises(ValueError):
        R.parse_slit_width("0")
    with pytest.raises(ValueError):
        R.parse_slit_width("xh=1")


def test_recipe_codes_cover_options():
    for field in ("temporal_src", "temporal_dst", "temporal_corridor", "temporal_extend", "calcarine_extend"):
        options = {"temporal_src": R.TEMPORAL_SRC_OPTIONS, "temporal_dst": R.TEMPORAL_DST_OPTIONS,
                   "temporal_corridor": R.TEMPORAL_CORRIDORS, "temporal_extend": R.TEMPORAL_EXTEND_OPTIONS,
                   "calcarine_extend": R.CALCARINE_EXTEND_OPTIONS}[field]
        for o in options:
            assert o in R.RECIPE_CODES[field]
