import importlib
from types import SimpleNamespace

import pytest
from PIL import Image

from src.generator import markdown_renderers as renderers
from src.generator.markdown_renderers import PlaywrightMarkdownRenderer, playwright_session


class FakePlaywright:
    """Records what the renderer asks of the Playwright sync API."""

    def __init__(self, fail_screenshots: int = 0):
        self.events: list[tuple] = []
        self.browsers: list[SimpleNamespace] = []
        self.fail_screenshots = fail_screenshots
        self.started = 0
        self.stopped = 0
        self.chromium = SimpleNamespace(launch=self._launch)

    def sync_playwright(self):
        return SimpleNamespace(start=self._start)

    def _start(self):
        self.started += 1
        return SimpleNamespace(chromium=self.chromium, stop=self._stop)

    def _stop(self):
        self.stopped += 1

    def _launch(self, headless, args):
        browser = SimpleNamespace(args=args, pages=[], closed=False)
        browser.new_page = lambda viewport, device_scale_factor: self._new_page(browser, viewport, device_scale_factor)
        browser.close = lambda: setattr(browser, "closed", True)
        self.browsers.append(browser)
        return browser

    def _new_page(self, browser, viewport, device_scale_factor):
        page = SimpleNamespace(viewport=dict(viewport), scale=device_scale_factor, gotos=0)
        browser.pages.append(page)

        def goto(url, wait_until):
            page.gotos += 1

        def screenshot(path, animations):
            if self.fail_screenshots > 0:
                self.fail_screenshots -= 1
                raise TimeoutError("renderer crashed")
            Image.new("RGB", (64, 96), "white").save(path)

        def set_viewport_size(size):
            page.viewport = dict(size)

        page.goto = goto
        page.wait_for_function = lambda script: None
        page.evaluate = lambda script: True
        page.locator = lambda selector: SimpleNamespace(screenshot=screenshot)
        page.set_viewport_size = set_viewport_size
        return page


@pytest.fixture
def fake(monkeypatch):
    fake_playwright = FakePlaywright()
    real_import_module = importlib.import_module

    def fake_import_module(name):
        if name == "playwright.sync_api":
            return SimpleNamespace(sync_playwright=fake_playwright.sync_playwright)
        return real_import_module(name)

    monkeypatch.setattr(renderers.importlib, "import_module", fake_import_module)
    return fake_playwright


def _renderer(scale: float = 1.0) -> PlaywrightMarkdownRenderer:
    renderer = PlaywrightMarkdownRenderer(font_path="/tmp/does-not-need-to-exist.ttf")
    renderer.style.render_scale = scale
    return renderer


def test_one_browser_and_page_serve_every_render_in_a_session(fake) -> None:
    renderer = _renderer()

    with playwright_session():
        for _ in range(5):
            renderer.render("# Title\n\nBody text")
        assert (fake.started, len(fake.browsers)) == (1, 1)
        assert fake.browsers[0].closed is False

    assert len(fake.browsers[0].pages) == 1
    assert fake.browsers[0].pages[0].gotos == 5
    assert fake.browsers[0].closed is True
    assert fake.stopped == 1


def test_the_viewport_follows_each_render(fake) -> None:
    renderer = _renderer()

    with playwright_session():
        renderer.render("short")
        renderer.render("\n\n".join(f"paragraph {i}" for i in range(400)))
        height = fake.browsers[0].pages[0].viewport["height"]

    assert height > 720


def test_each_render_scale_gets_its_own_browser_with_the_same_launch_flags(fake) -> None:
    with playwright_session():
        _renderer(1.0).render("a")
        _renderer(2.0).render("b")
        _renderer(1.0).render("c")

    assert [b.args[-1] for b in fake.browsers] == [
        "--force-device-scale-factor=1.0",
        "--force-device-scale-factor=2.0",
    ]
    assert fake.browsers[0].pages[0].gotos == 2


def test_the_least_recently_used_scale_is_closed_when_too_many_are_open(fake) -> None:
    with playwright_session():
        for scale in (1.0, 2.0, 3.0):
            _renderer(scale).render("x")
        assert [b.closed for b in fake.browsers] == [True, False, False]


def test_the_page_is_replaced_after_recycle_after_renders(fake) -> None:
    renderer = _renderer()

    with playwright_session(recycle_after=3):
        for _ in range(7):
            renderer.render("x")

    assert len(fake.browsers) == 3  # renders 1-3, 4-6, 7
    assert [b.pages[0].gotos for b in fake.browsers] == [3, 3, 1]
    assert all(b.closed for b in fake.browsers)


def test_render_error_is_wrapped_and_the_broken_browser_is_closed(fake) -> None:
    renderer = _renderer()

    with playwright_session():
        renderer.render("ok")
        fake.fail_screenshots = 1
        with pytest.raises(RuntimeError, match="Headless Playwright markdown rendering failed"):
            renderer.render("boom")
        assert fake.browsers[0].closed is True
        renderer.render("recovered")
        assert len(fake.browsers) == 2
        assert fake.browsers[1].closed is False


def test_rendering_outside_a_session_still_launches_and_closes_a_browser_per_call(fake, monkeypatch) -> None:
    class OneShot:
        def __init__(self, playwright):
            self.playwright = playwright

        def __enter__(self):
            return SimpleNamespace(chromium=self.playwright.chromium)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(fake, "sync_playwright", lambda: OneShot(fake))
    renderer = _renderer()

    renderer.render("first")
    renderer.render("second")

    assert len(fake.browsers) == 2
    assert all(browser.closed for browser in fake.browsers)
    assert fake.started == 0


def test_sessions_nest_and_restore_the_previous_one(fake) -> None:
    assert renderers._active_session is None
    with playwright_session() as outer:
        assert renderers._active_session is outer
        with playwright_session() as inner:
            assert renderers._active_session is inner
        assert renderers._active_session is outer
    assert renderers._active_session is None
