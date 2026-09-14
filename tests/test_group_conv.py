"""Tests for D2 Group Convolution and Backbone encoder."""

import torch
from soccer_ml.data.transforms import D2_MULTIPLY
from soccer_ml.models.group_conv import D2GroupConvLayer
from soccer_ml.models.backbone import D2GATv2Backbone


def test_d2_group_conv_shapes():
    """Verify group conv returns 4 views of expected shape."""
    b, n, in_dim, out_dim = 2, 22, 14, 32
    layer = D2GroupConvLayer(
        in_node_dim=in_dim,
        out_node_dim=out_dim,
        nb_heads=4,
        in_edge_dim=2,
        in_graph_dim=4,
    )

    node_views = [torch.randn(b, n, in_dim) for _ in range(4)]
    edge_views = [torch.randn(b, n, n, 2) for _ in range(4)]
    graph_views = [torch.randn(b, 4) for _ in range(4)]

    out_views = layer(node_views, edge_views, graph_views)
    assert len(out_views) == 4
    for v in out_views:
        assert v.shape == (b, n, out_dim)


def test_d2_group_equivariance():
    """Verify D2 group equivariance of group convolution.

    If input views are shifted by action k (node_views[g] = orig_views[k * g]),
    the output views must satisfy out_views[g] = orig_out_views[k * g].
    """
    torch.manual_seed(42)
    b, n, in_dim, out_dim = 2, 10, 8, 16
    layer = D2GroupConvLayer(
        in_node_dim=in_dim,
        out_node_dim=out_dim,
        nb_heads=2,
        in_edge_dim=1,
        in_graph_dim=2,
        dropout=0.0,
    )
    layer.eval()

    orig_node_views = [torch.randn(b, n, in_dim) for _ in range(4)]
    orig_edge_views = [torch.randn(b, n, n, 1) for _ in range(4)]
    orig_graph_views = [torch.randn(b, 2) for _ in range(4)]

    orig_out_views = layer(orig_node_views, orig_edge_views, orig_graph_views)

    # Test for action k in D2
    for k in [1, 2, 3]:
        # Permute input views: view_permuted[g] = orig[k * g]
        perm_node_views = [orig_node_views[D2_MULTIPLY[k][g]] for g in range(4)]
        perm_edge_views = [orig_edge_views[D2_MULTIPLY[k][g]] for g in range(4)]
        perm_graph_views = [orig_graph_views[D2_MULTIPLY[k][g]] for g in range(4)]

        perm_out_views = layer(perm_node_views, perm_edge_views, perm_graph_views)

        for g in range(4):
            expected_view = orig_out_views[D2_MULTIPLY[k][g]]
            assert torch.allclose(perm_out_views[g], expected_view, atol=1e-5), (
                f"Equivariance broken for action {k} at view {g}"
            )


def test_backbone_forward():
    """Verify D2GATv2Backbone forward pass end-to-end."""
    b, n, in_dim, hidden_dim = 2, 22, 7, 32
    backbone = D2GATv2Backbone(
        in_node_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        nb_heads=4,
        in_edge_dim=1,
        in_graph_dim=4,
    )

    node_fts = torch.randn(b, n, in_dim)
    edge_fts = torch.randn(b, n, n, 1)
    graph_fts = torch.randn(b, 4)

    out_views = backbone(node_fts, edge_fts, graph_fts)
    assert len(out_views) == 4
    for v in out_views:
        assert v.shape == (b, n, hidden_dim)
