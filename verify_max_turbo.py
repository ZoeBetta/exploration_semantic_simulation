"""Regression checks for the v32 Max Turbo viewer mode.

The test prefers the installed pygame-ce package with SDL's dummy driver.  On a
source-review machine without pygame-ce, it installs a tiny in-memory stand-in
that exercises the same Turbo control flow without rendering pixels.
"""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gui as gui_module  # noqa: E402


class _FakeRect:
    def __init__(self, left, top, width, height):
        self.left = int(left)
        self.top = int(top)
        self.width = int(width)
        self.height = int(height)

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def center(self):
        return self.left + self.width // 2, self.top + self.height // 2

    @property
    def topleft(self):
        return self.left, self.top

    @property
    def size(self):
        return self.width, self.height

    def colliderect(self, other):
        return not (
            self.right <= other.left or other.right <= self.left or
            self.bottom <= other.top or other.bottom <= self.top
        )

    def collidepoint(self, point):
        x, y = point
        return self.left <= x < self.right and self.top <= y < self.bottom


class _FakeSurface:
    def __init__(self, size, _flags=0):
        self._size = tuple(map(int, size))

    def get_size(self):
        return self._size

    def get_width(self):
        return self._size[0]

    def get_height(self):
        return self._size[1]

    def copy(self):
        return _FakeSurface(self._size)

    def fill(self, _colour):
        return None

    def blit(self, _surface, _where):
        return None

    def get_rect(self, **kwargs):
        rect = _FakeRect(0, 0, self._size[0], self._size[1])
        if "center" in kwargs:
            cx, cy = kwargs["center"]
            rect.left = int(cx - rect.width / 2)
            rect.top = int(cy - rect.height / 2)
        return rect


class _FakeFont:
    @staticmethod
    def render(text, _antialias, _colour):
        return _FakeSurface((max(1, len(str(text)) * 7), 18))


class _FakeClock:
    @staticmethod
    def tick(_fps=0):
        return 0


class _FakeDisplay:
    def __init__(self):
        self.surface = None

    @staticmethod
    def get_desktop_sizes():
        return [(1200, 800)]

    def get_surface(self):
        return self.surface

    def set_mode(self, size):
        self.surface = _FakeSurface(size)
        return self.surface

    @staticmethod
    def set_caption(_caption):
        return None

    @staticmethod
    def update(_region=None):
        return None

    def quit(self):
        self.surface = None


class _FakePygame:
    SRCALPHA = 1
    QUIT = 10
    MOUSEBUTTONDOWN = 11
    KEYDOWN = 12
    K_ESCAPE = 27
    K_SPACE = 32
    K_1, K_2, K_3, K_4 = 49, 50, 51, 52
    Rect = _FakeRect
    Surface = _FakeSurface

    def __init__(self):
        self._initialised = False
        self.display = _FakeDisplay()
        self.font = SimpleNamespace(SysFont=lambda *_a, **_k: _FakeFont())
        self.time = SimpleNamespace(Clock=lambda: _FakeClock())
        self.event = SimpleNamespace(get=lambda: [])
        self.draw = SimpleNamespace(rect=lambda *_a, **_k: None)

    def init(self):
        self._initialised = True

    def get_init(self):
        return self._initialised

    def quit(self):
        self._initialised = False


if gui_module.pygame is None:
    gui_module.pygame = _FakePygame()

pygame = gui_module.pygame
RunViewer = gui_module.RunViewer


class _Robot:
    floor = 0


class _DummyState:
    def __init__(self, target_steps: int):
        self.target_steps = int(target_steps)
        self.steps = 0
        self.finished = False
        self.robot = _Robot()
        self.floors = []

    def step(self, _dt: float) -> bool:
        self.steps += 1
        if self.steps >= self.target_steps:
            self.finished = True
            return False
        return True

    @staticmethod
    def interpolated_pose(_alpha: float):
        return 0.0, 0.0, 0.0


def main() -> None:
    turbo_rect = RunViewer._max_turbo_button_rect(1500)
    camera_rect = RunViewer._camera_button_rect(1500)
    assert not turbo_rect.colliderect(camera_rect)
    assert 0 <= camera_rect.left - turbo_rect.right <= 16

    RunViewer._persistent_time_scale = 4.0
    RunViewer._persistent_max_turbo = True
    RunViewer._persistent_frozen_frame = None

    first_state = _DummyState(1200)
    first = RunViewer(first_state, "Run 1/2 (episodio 1/2)")
    first._draw_frame = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("full rendering must not run in Max Turbo")
    )
    started = time.perf_counter()
    returned_speed = first.run()
    elapsed = time.perf_counter() - started

    assert first_state.steps == 1200
    assert returned_speed == 4.0
    assert RunViewer._persistent_max_turbo is True
    assert elapsed < 5.0, "dummy Max Turbo run unexpectedly throttled"

    second_state = _DummyState(300)
    second = RunViewer(second_state, "Run 2/2 (episodio 2/2)")
    assert second.max_turbo is True
    second._draw_frame = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("full rendering must not run in inherited Max Turbo")
    )
    second.run()
    assert second_state.steps == 300
    assert second.time_scale == 4.0

    screen = pygame.display.get_surface()
    assert screen is not None
    second._set_max_turbo(False, screen, (0.0, 0.0, 0.0))
    assert second.max_turbo is False
    assert RunViewer._persistent_max_turbo is False
    assert second.time_scale == 4.0

    RunViewer.shutdown_display()
    print("Max Turbo regression checks passed.")


if __name__ == "__main__":
    main()
