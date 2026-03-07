from PIL import Image

import generator.markdown_render_utils as render_utils



def test_formula_cache_returns_copy_and_avoids_rerender(monkeypatch) -> None:
    render_utils._FORMULA_IMAGE_CACHE.clear()
    monkeypatch.setattr(render_utils, "_FORMULA_IMAGE_CACHE_MAX_ITEMS", 4)

    render_calls = []

    monkeypatch.setattr(render_utils, "_get_latex_to_image_renderer", lambda: object())
    monkeypatch.setattr(
        render_utils,
        "_render_formula_array_with_latex_tools",
        lambda _renderer, expression: render_calls.append(expression) or [[0, 255], [255, 0]],
    )
    monkeypatch.setattr(
        render_utils,
        "_formula_array_to_rgba",
        lambda _array, _color: Image.new("RGBA", (12, 8), (255, 255, 255, 255)),
    )

    first = render_utils.render_formula_image("x^2", 18, (0, 0, 0))
    second = render_utils.render_formula_image("x^2", 18, (0, 0, 0))

    assert first is not None
    assert second is not None
    assert render_calls == ["x^2"]
    assert first is not second



def test_formula_cache_evicts_oldest_entries(monkeypatch) -> None:
    render_utils._FORMULA_IMAGE_CACHE.clear()
    monkeypatch.setattr(render_utils, "_FORMULA_IMAGE_CACHE_MAX_ITEMS", 2)

    monkeypatch.setattr(render_utils, "_get_latex_to_image_renderer", lambda: object())
    monkeypatch.setattr(
        render_utils,
        "_render_formula_array_with_latex_tools",
        lambda _renderer, _expression: [[0, 255], [255, 0]],
    )
    monkeypatch.setattr(
        render_utils,
        "_formula_array_to_rgba",
        lambda _array, _color: Image.new("RGBA", (10, 10), (255, 255, 255, 255)),
    )

    render_utils.render_formula_image("a", 16, (0, 0, 0))
    render_utils.render_formula_image("b", 16, (0, 0, 0))
    render_utils.render_formula_image("c", 16, (0, 0, 0))

    cache_keys = list(render_utils._FORMULA_IMAGE_CACHE.keys())
    assert len(cache_keys) == 2
    assert cache_keys[0][0] == "b"
    assert cache_keys[1][0] == "c"
