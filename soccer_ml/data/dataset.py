"""PyTorch Dataset implementations, synthetic data generator, and collation for soccer graphs."""

from typing import List, Optional
import torch
from torch.utils.data import Dataset

from soccer_ml.data.schema import SoccerSample, SoccerGraphBatch


def collate_soccer_graphs(batch: List[SoccerSample]) -> SoccerGraphBatch:
    """Collates a list of SoccerSample instances into a single SoccerGraphBatch."""
    node_fts = torch.stack([s.node_features for s in batch], dim=0)  # (B, N, D_v)
    edge_fts = torch.stack([s.edge_features for s in batch], dim=0)  # (B, N, N, D_e)
    graph_fts = torch.stack([s.graph_features for s in batch], dim=0)  # (B, D_g)
    adj_mat = torch.stack([s.adj_mat for s in batch], dim=0)  # (B, N, N)

    # Optional labels
    receiver_label = (
        torch.stack([s.receiver_label for s in batch], dim=0)
        if batch[0].receiver_label is not None
        else None
    )
    target_zone_label = (
        torch.stack([s.target_zone_label for s in batch], dim=0)
        if batch[0].target_zone_label is not None
        else None
    )
    pass_end_location = (
        torch.stack([s.pass_end_location for s in batch], dim=0)
        if batch[0].pass_end_location is not None
        else None
    )
    shot_label = (
        torch.stack([s.shot_label for s in batch], dim=0)
        if batch[0].shot_label is not None
        else None
    )
    creativity_label = (
        torch.stack([s.creativity_label for s in batch], dim=0)
        if batch[0].creativity_label is not None
        else None
    )
    player_mask = (
        torch.stack([s.player_mask for s in batch], dim=0)
        if batch[0].player_mask is not None
        else None
    )

    return SoccerGraphBatch(
        node_features=node_fts,
        edge_features=edge_fts,
        graph_features=graph_fts,
        adj_mat=adj_mat,
        receiver_label=receiver_label,
        target_zone_label=target_zone_label,
        pass_end_location=pass_end_location,
        shot_label=shot_label,
        creativity_label=creativity_label,
        player_mask=player_mask,
    )


class SoccerGraphDataset(Dataset):
    """Generic dataset wrapping a list of SoccerSample objects."""

    def __init__(self, samples: List[SoccerSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SoccerSample:
        return self.samples[idx]


class SyntheticSoccerDataset(Dataset):
    """Generates synthetic soccer tracking frames mimicking StatsBomb 360 / TacticAI data.

    Useful for pipeline verification, test suites, and benchmarking before real data parsing.
    """

    def __init__(
        self,
        num_samples: int = 128,
        num_players: int = 22,
        node_dim: int = 7,
        edge_dim: int = 1,
        graph_dim: int = 4,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_players = num_players
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.graph_dim = graph_dim

        if seed is not None:
            torch.manual_seed(seed)

        self.samples: List[SoccerSample] = []
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        n = self.num_players

        # Build team membership: first n//2 attacking team (0), second n//2 defending team (1)
        team_ids = torch.zeros(n, dtype=torch.long)
        team_ids[n // 2 :] = 1

        # Edge feature: 1.0 if opposing team, 0.0 if same team
        edge_fts = (team_ids.unsqueeze(0) != team_ids.unsqueeze(1)).float().unsqueeze(-1)  # (N, N, 1)
        if self.edge_dim > 1:
            extra_edges = torch.zeros(n, n, self.edge_dim - 1)
            edge_fts = torch.cat([edge_fts, extra_edges], dim=-1)

        # Fully connected adjacency matrix
        adj_mat = torch.ones(n, n, dtype=torch.float32)

        for _ in range(self.num_samples):
            # Coordinates in [-5.0, 5.0] (10m x 10m normalized box)
            xy = (torch.rand(n, 2) - 0.5) * 10.0

            # Velocities in [-3.0, 3.0] m/s
            v_xy = (torch.rand(n, 2) - 0.5) * 6.0

            # Height (normalized ~1.80) and weight (normalized ~0.75)
            height = torch.randn(n, 1) * 0.07 + 1.82
            weight = torch.randn(n, 1) * 0.08 + 0.77

            # Ball possession: one player has ball
            ball_possession = torch.zeros(n, 1)
            ball_player = torch.randint(0, n, (1,)).item()
            ball_possession[ball_player, 0] = 1.0

            node_fts = torch.cat([xy, v_xy, height, weight, ball_possession], dim=-1)
            if self.node_dim > 7:
                extra_nodes = torch.zeros(n, self.node_dim - 7)
                node_fts = torch.cat([node_fts, extra_nodes], dim=-1)

            # Global features
            graph_fts = torch.randn(self.graph_dim)

            # Target labels:
            # 1. Receiver prediction (0 to N-1)
            receiver_label = torch.randint(0, n, ()).long()

            # 2. Shot prediction (0.0 or 1.0)
            shot_label = torch.tensor([1.0 if torch.rand(()) > 0.6 else 0.0], dtype=torch.float32)

            # 3. Continuous player rating / creativity score (N,)
            creativity_label = torch.randn(n, dtype=torch.float32)

            player_mask = torch.ones(n, dtype=torch.bool)

            self.samples.append(
                SoccerSample(
                    node_features=node_fts,
                    edge_features=edge_fts.clone(),
                    graph_features=graph_fts,
                    adj_mat=adj_mat.clone(),
                    receiver_label=receiver_label,
                    shot_label=shot_label,
                    creativity_label=creativity_label,
                    player_mask=player_mask,
                )
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> SoccerSample:
        return self.samples[idx]
