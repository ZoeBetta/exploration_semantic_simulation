"""Regression tests for planner-starvation recovery and semantic legend groups."""
from __future__ import annotations

import inspect
import numpy as np

import gui
import planner
import simulation
from robot import FloorMap


def test_relaxed_frontier_mask_is_superset():
    """Relaxation may add candidates but must never remove strict candidates."""
    fmap = FloorMap()
    # A small synthetic observed rectangle leaves a clean free/unknown border.
    fmap.visibility_observed[80:140, 80:180] = True
    strict = planner.detect_standard_frontiers(fmap, relaxed=False)
    relaxed = planner.detect_standard_frontiers(fmap, relaxed=True)
    assert len(relaxed) >= len(strict)


def test_starvation_command_rotates_in_place():
    """Source-level guard against the old silent (0, 0) freeze branch."""
    source = inspect.getsource(simulation.RunState.step)
    assert 'PLANNER_STARVATION_TURN_RAD_S' in source
    assert 'if self.planner_starved else 0.0' in source


def test_legend_semantic_groups():
    groups = gui.RunViewer._legend_groups()
    names = [name for name, _ in groups]
    assert names == [
        'AMBIENTE REALE',
        'ROBOT E SENSORI',
        'PIANIFICAZIONE E MOVIMENTO',
        'MAPPE / STRUTTURE DATI',
    ]
    flattened = [label for _, entries in groups for label, _, _ in entries]
    assert 'Percorso A*' in flattened
    assert 'Mappa 5 cm: occupato' in flattened
    assert 'Scala SU' in flattened


if __name__ == '__main__':
    test_relaxed_frontier_mask_is_superset()
    test_starvation_command_rotates_in_place()
    test_legend_semantic_groups()
    print('Frontier recovery and semantic legend checks: OK')
