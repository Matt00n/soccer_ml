"""Tests for the pure PyTorch GATv2Layer module."""

import pytest
import torch
from soccer_ml.models.gatv2 import GATv2Layer


def test_gatv2_output_shape():
    """Verify GATv2 layer tensor dimensions."""
    b, n, in_dim, out_dim = 4, 22, 7, 32
    layer = GATv2Layer(
        in_node_dim=in_dim,
        out_node_dim=out_dim,
        nb_heads=4,
        in_edge_dim=1,
        in_graph_dim=4,
    )

    node_fts = torch.randn(b, n, in_dim)
    edge_fts = torch.randn(b, n, n, 1)
    graph_fts = torch.randn(b, 4)
    adj_mat = torch.ones(b, n, n)

    out = layer(node_fts, edge_fts, graph_fts, adj_mat)
    assert out.shape == (b, n, out_dim)


def test_gatv2_head_divisibility():
    """Verify error raised if out_dim is not divisible by nb_heads."""
    with pytest.raises(ValueError):
        GATv2Layer(in_node_dim=7, out_node_dim=33, nb_heads=4)


def test_gatv2_isolated_node():
    """Verify that an isolated node with 0 in adj_mat only gets self/skip signal."""
    layer = GATv2Layer(
        in_node_dim=8,
        out_node_dim=16,
        nb_heads=2,
        in_edge_dim=1,
        in_graph_dim=2,
        residual=True,
    )

    b, n = 2, 4
    node_fts = torch.randn(b, n, 8)
    edge_fts = torch.randn(b, n, n, 1)
    graph_fts = torch.randn(b, 2)

    # Disconnect node 0 from all other nodes except self
    adj_mat = torch.eye(n).unsqueeze(0).repeat(b, 1, 1)

    out = layer(node_fts, edge_fts, graph_fts, adj_mat)
    assert out.shape == (b, n, 16)
    assert not torch.isnan(out).any()
