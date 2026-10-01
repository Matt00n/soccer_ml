"""Tests for the temporal pass-target dataset on the toy StatsBomb 360 match."""

from pathlib import Path

import pytest
import torch

from soccer_ml.data.pitch_grid import PitchGrid
from soccer_ml.data.temporal_receiver_dataset import (
    EVENT_END_X_IDX,
    EVENT_END_Y_IDX,
    EVENT_SAME_TEAM_IDX,
    EVENT_X_IDX,
    EVENT_Y_IDX,
    TARGET_PASS_ATTRIBUTE_INDICES,
    TemporalReceiverDataset,
    collate_temporal_receiver_graphs,
)

THREE_SIXTY = Path("data_toy_receiver/three_sixty_3764440.json")
EVENTS = Path("data_toy_receiver/events_3764440.json")

pytestmark = pytest.mark.skipif(not THREE_SIXTY.exists(), reason="toy StatsBomb data not available")


@pytest.fixture(scope="module")
def dataset() -> TemporalReceiverDataset:
    return TemporalReceiverDataset(THREE_SIXTY, EVENTS, context_length=8, max_samples=300)


def test_zone_labels_match_pass_end_locations(dataset):
    grid = PitchGrid(26, 17)
    batch = collate_temporal_receiver_graphs([dataset[i] for i in range(len(dataset))])
    assert batch.target_zone_label.max().item() < grid.num_zones
    torch.testing.assert_close(grid.batch_coords_to_zones(batch.pass_end_location), batch.target_zone_label)


def test_target_pass_attributes_are_hidden_by_default(dataset):
    batch = collate_temporal_receiver_graphs([dataset[i] for i in range(len(dataset))])
    assert torch.all(batch.graph_features[:, -1, list(TARGET_PASS_ATTRIBUTE_INDICES)] == 0)
    with_attributes = TemporalReceiverDataset(
        THREE_SIXTY, EVENTS, context_length=2, max_samples=50, include_target_pass_attributes=True
    )
    target_graphs = torch.stack([sample.graph_features[-1] for sample in with_attributes.samples])
    assert target_graphs[:, TARGET_PASS_ATTRIBUTE_INDICES[0]].abs().sum() > 0


def test_history_is_in_target_team_frame(dataset):
    """Opponent events are rotated into the passer's frame.

    Their actor is then not a teammate of the passer, their location agrees
    with the actor's freeze-frame position, and they are spatially continuous
    with the preceding event of the passer's own team.  Medians are used
    because some StatsBomb frames (e.g. opponent ball receipts) flag an actor
    that does not stand at the event location even in the raw data.
    """
    actor_offsets, continuity_gaps = [], []
    for sample in dataset.samples:
        if not sample.metadata["history_opponent_events"]:
            continue
        valid_steps = sample.sequence_mask.nonzero(as_tuple=True)[0][:-1].tolist()
        for previous, step in zip([None] + valid_steps, valid_steps):
            if sample.event_numeric[step, EVENT_SAME_TEAM_IDX] > 0.5:
                continue
            actor = sample.role_features[step, :, 0].argmax()
            assert sample.role_features[step, actor, 2] == 0
            assert sample.node_features[step, actor, 7] == 0
            actor_xy = sample.node_features[step, actor, :2] / 10.0 + 0.5
            event_xy = sample.event_numeric[step, [EVENT_X_IDX, EVENT_Y_IDX]]
            actor_offsets.append(torch.norm(actor_xy - event_xy).item())
            if previous is not None and sample.event_numeric[previous, EVENT_SAME_TEAM_IDX] > 0.5:
                previous_end = sample.event_numeric[previous, [EVENT_END_X_IDX, EVENT_END_Y_IDX]]
                continuity_gaps.append(torch.norm(event_xy - previous_end).item())
    assert len(actor_offsets) > 20 and len(continuity_gaps) > 10
    assert torch.tensor(actor_offsets).median() < 0.02
    # Unrotated, an opponent pressure sits on the mirrored side of the pitch.
    assert torch.tensor(continuity_gaps).median() < 0.1
