# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import typing

import pytest

from geolangsplat.config import (
    RECIPES,
    GeoLangSplatConfig,
    RecipeName,
    apply_recipe,
    config_field_names,
    list_recipes,
)
from geolangsplat.errors import GeoLangSplatError
from geolangsplat.cli._common import build_config


def test_recipe_name_matches_recipes():
    # the CLI choice type must stay in sync with the actual recipe registry
    assert set(typing.get_args(RecipeName)) == set(RECIPES)


def test_recipes_exist():
    names = list_recipes()
    assert {"aerial", "satellite", "satellite_dense"} <= set(names)


def test_competition_off_by_default():
    cfg = GeoLangSplatConfig()
    assert cfg.compete is False
    assert len(cfg.distractors) > 0  # a generic competition set is still available


def test_recipes_do_not_force_competition():
    # recipes supply curated distractors but must not silently turn competition on
    for name in list_recipes():
        cfg = GeoLangSplatConfig()
        apply_recipe(cfg, name)
        assert cfg.compete is False


def test_apply_recipe_fills_defaults():
    cfg = GeoLangSplatConfig()
    applied = apply_recipe(cfg, "satellite")
    assert "select" in applied
    assert cfg.select == 0.35
    assert cfg.distractors == ("road", "grass", "tree", "water")


def test_explicit_value_wins_over_preset():
    cfg = GeoLangSplatConfig(select=0.99)
    applied = apply_recipe(cfg, "satellite")
    assert "select" not in applied  # not overwritten
    assert cfg.select == 0.99


def test_apply_recipe_none_is_noop():
    cfg = GeoLangSplatConfig()
    assert apply_recipe(cfg, None) == []


def test_unknown_recipe_raises():
    with pytest.raises(GeoLangSplatError):
        apply_recipe(GeoLangSplatConfig(), "nope")


def test_unknown_recipe_suggests_closest():
    with pytest.raises(GeoLangSplatError, match="did you mean 'satellite'"):
        apply_recipe(GeoLangSplatConfig(), "satelite")


def test_build_config_ignores_none():
    cfg = build_config(select=None, margin=0.2, device="cpu")
    assert cfg.select == GeoLangSplatConfig().select  # untouched
    assert cfg.margin == 0.2
    assert cfg.device == "cpu"


def test_config_field_names_nonempty():
    names = config_field_names()
    assert "select" in names and "view_source" in names and "lift" in names
