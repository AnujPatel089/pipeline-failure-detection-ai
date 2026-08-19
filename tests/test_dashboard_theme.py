import pytest

from src.dashboard_styles import DEFAULT_THEME, REQUIRED_THEME_KEYS, THEME_NAMES, THEMES, get_theme


def _hex_to_luminance(value: str) -> float:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def test_all_four_themes_exist() -> None:
    assert THEME_NAMES == ["Industrial Slate", "Deep Navy", "Light Operations", "Steel Blue"]
    assert set(THEMES) == set(THEME_NAMES)


def test_industrial_slate_is_default() -> None:
    assert DEFAULT_THEME == "Industrial Slate"
    assert get_theme(None) is THEMES["Industrial Slate"]


@pytest.mark.parametrize("theme_name", ["Industrial Slate", "Deep Navy", "Light Operations", "Steel Blue"])
def test_theme_has_every_required_semantic_key(theme_name: str) -> None:
    theme = THEMES[theme_name]
    assert REQUIRED_THEME_KEYS.issubset(theme.keys())
    assert all(isinstance(value, str) and value for value in theme.values())


@pytest.mark.parametrize("invalid_name", [None, "", "Nonexistent Theme", "industrial slate"])
def test_invalid_theme_falls_back_to_default(invalid_name: str | None) -> None:
    assert get_theme(invalid_name) == THEMES[DEFAULT_THEME]


def test_light_operations_uses_light_background_and_dark_text() -> None:
    theme = THEMES["Light Operations"]
    assert _hex_to_luminance(theme["background"]) > 200
    assert _hex_to_luminance(theme["text_primary"]) < 80


@pytest.mark.parametrize("theme_name", ["Industrial Slate", "Deep Navy", "Steel Blue"])
def test_dark_themes_use_light_readable_primary_text(theme_name: str) -> None:
    theme = THEMES[theme_name]
    assert _hex_to_luminance(theme["background"]) < 60
    assert _hex_to_luminance(theme["text_primary"]) > 200
