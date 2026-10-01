"""Tests for D2 group transformations and pitch coordinate normalization."""

import torch
from soccer_ml.data.transforms import (
    D2GroupTransform,
    D2_MULTIPLY,
    D2_INVERSES,
    normalize_pitch_coordinates,
)


def test_d2_self_inverses():
    """Verify that every member of D2 is a self-inverse: g(g(x)) == x."""
    transform = D2GroupTransform()
    x = torch.randn(4, 22, 7)

    for action in range(4):
        # g(g(x))
        x_once = transform.transform_node_features(x, action)
        x_twice = transform.transform_node_features(x_once, action)
        assert torch.allclose(x, x_twice, atol=1e-6), f"Action {action} is not a self-inverse!"


def test_d2_cayley_multiplication():
    """Verify that group composition g * h satisfies D2 multiplication table."""
    transform = D2GroupTransform()
    x = torch.randn(4, 22, 7)

    for g in range(4):
        for h in range(4):
            gh = D2_MULTIPLY[g][h]

            # Apply h then g: (g * h)(x)
            x_h = transform.transform_node_features(x, h)
            x_gh_seq = transform.transform_node_features(x_h, g)

            # Apply direct composite gh
            x_gh_direct = transform.transform_node_features(x, gh)

            assert torch.allclose(x_gh_seq, x_gh_direct, atol=1e-6), (
                f"Failed composition for g={g}, h={h} -> expected {gh}"
            )


def test_pitch_normalization():
    """Verify pitch zero-centering and target box scaling."""
    x = torch.tensor([0.0, 52.5, 105.0])
    y = torch.tensor([0.0, 34.0, 68.0])

    norm_x, norm_y = normalize_pitch_coordinates(x, y, pitch_length=105.0, pitch_width=68.0, target_scale=10.0)

    # Center (52.5, 34.0) should map to (0.0, 0.0)
    assert abs(norm_x[1].item()) < 1e-5
    assert abs(norm_y[1].item()) < 1e-5

    # Bounds should be in [-5.0, 5.0]
    assert abs(norm_x[0].item() - (-5.0)) < 1e-5
    assert abs(norm_x[2].item() - 5.0) < 1e-5
    assert abs(norm_y[0].item() - (-5.0)) < 1e-5
    assert abs(norm_y[2].item() - 5.0) < 1e-5


def test_statsbomb_layout_reflections_only_touch_coordinates():
    """StatsBomb nodes have no velocities: flags must survive every reflection."""
    from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT

    transform = D2GroupTransform.from_layout(STATSBOMB_NODE_LAYOUT)
    nodes = torch.tensor([[[1.0, 2.0, 1.0, 1.0, 0.0], [-3.0, 0.5, 0.0, 0.0, 1.0]]])
    expected = {
        1: torch.tensor([[[-1.0, 2.0, 1.0, 1.0, 0.0], [3.0, 0.5, 0.0, 0.0, 1.0]]]),
        2: torch.tensor([[[1.0, -2.0, 1.0, 1.0, 0.0], [-3.0, -0.5, 0.0, 0.0, 1.0]]]),
        3: torch.tensor([[[-1.0, -2.0, 1.0, 1.0, 0.0], [3.0, -0.5, 0.0, 0.0, 1.0]]]),
    }
    for action, target in expected.items():
        torch.testing.assert_close(transform.transform_node_features(nodes, action), target)
