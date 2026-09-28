"""Route keyword arguments handed to ``Hemi.seam_paths`` must keep numeric fields numeric."""
from __future__ import annotations

import pytest

import mfa.recipe as R
from mfa.recipe import NUMERIC_ROUTE_FIELDS, ROUTE_FIELDS, default_recipe, parse_recipe, route_kwargs


def test_border_pull_stays_a_float() -> None:
    """Regression: ``border_pull`` was stringified, which made anchored routes crash."""
    kw = route_kwargs(default_recipe())
    assert isinstance(kw["border_pull"], float)


def test_every_numeric_route_field_is_numeric() -> None:
    kw = route_kwargs(default_recipe())
    for field in NUMERIC_ROUTE_FIELDS:
        if field in ROUTE_FIELDS:
            assert isinstance(kw[field], float), field


def test_anchored_recipe_round_trip_keeps_pull() -> None:
    r = parse_recipe("frontal_route=border:OFC_ACgG,frontal_corridor=pair,"
                     "frontal_src=nearest_OFC_ACgG_border,frontal_dst=OFC_far_ACgG,border_pull=0.25")
    kw = route_kwargs(r)
    assert kw["border_pull"] == pytest.approx(0.25)
    assert kw["frontal_route"] == "border:OFC_ACgG"


def test_option_fields_are_strings() -> None:
    kw = route_kwargs(default_recipe())
    for field in ROUTE_FIELDS:
        if field not in NUMERIC_ROUTE_FIELDS:
            assert isinstance(kw[field], str), field


def test_border_route_values_are_coordinate_free():
    """A route mode's implied endpoint names are positions on a named border, not coordinates.

    They were once rejected by the coordinate-free guard, which made every `border_trace`
    recipe impossible to load even though such a recipe names no axis at all.
    """
    for field, value in (("temporal_src", R.BORDER_RIM), ("temporal_dst", R.BORDER_END),
                         ("temporal_corridor", R.BAND), ("frontal_corridor", R.PAIR),
                         ("temporal_route", "border_trace:Amy_TG"), ("frontal_route", "border:OFC_ACgG")):
        assert R.is_coordinate_free(field, value), (field, value)


def test_border_trace_recipe_loads():
    r = R.parse_recipe("temporal_route=border_trace:Amy_TG,temporal_src=border_rim,"
                       "temporal_corridor=band,temporal_dst=border_end")
    assert r["temporal_src"] == R.BORDER_RIM and r["temporal_dst"] == R.BORDER_END


def test_seam_paths_defaults_are_one_consistent_recipe(synthetic_subject):
    """A bare `seam_paths` call must use the packaged recipe for EVERY route field.

    It used to take the corridor from the recipe but the route mode from a hard-coded
    default, which mixes two recipes: with a packaged recipe whose frontal cut traces a
    border, that combination raised `unknown frontal corridor 'band'`.
    """
    import os

    from mfa.surface import Hemi
    subjects_dir, subject = synthetic_subject
    H = Hemi(os.path.join(subjects_dir, subject), "lh")
    bare = H.seam_paths(("calcarine", "frontal", "temporal"))
    explicit = H.seam_paths(("calcarine", "frontal", "temporal"), **route_kwargs(default_recipe()))
    assert set(bare) == set(explicit)
    for k in bare:
        assert (bare[k] == explicit[k]).all(), k
