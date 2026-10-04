"""Match-level train/validation splits for StatsBomb 360 data directories."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset


def match_id_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ("three_sixty_", "three-sixty_", "events_", "event_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    digits = "".join(character for character in stem if character.isdigit())
    return digits or stem


def find_match_pairs(data_dir: str, sample_count: int | None, seed: int) -> list[tuple[str, str]]:
    """Pair three-sixty and event files, optionally sampling ``sample_count`` matches.

    Supports the StatsBomb open-data layout (``three-sixty/`` and ``events/``
    subfolders) and flat folders with ``three_sixty_<id>.json``/``events_<id>.json``.
    """
    root = Path(data_dir)
    three_sixty_dir, events_dir = root / "three-sixty", root / "events"
    pairs: list[tuple[str, str]] = []
    if three_sixty_dir.exists() and events_dir.exists():
        for freeze_frame_path in sorted(three_sixty_dir.glob("*.json")):
            event_path = events_dir / f"{freeze_frame_path.stem}.json"
            if event_path.exists():
                pairs.append((str(freeze_frame_path), str(event_path)))
    else:
        three_sixty_map: dict[str, str] = {}
        events_map: dict[str, str] = {}
        for path in sorted(root.glob("*.json")):
            match_id = match_id_from_filename(path)
            name = path.name.lower()
            if "event" in name:
                events_map[match_id] = str(path)
            elif "three" in name or "sixty" in name:
                three_sixty_map[match_id] = str(path)
            else:
                three_sixty_map.setdefault(match_id, str(path))
                events_map.setdefault(match_id, str(path))
        pairs = [(three_sixty_map[key], events_map[key]) for key in sorted(three_sixty_map.keys() & events_map.keys())]
    if not pairs:
        raise FileNotFoundError(f"Could not match three-sixty/events JSON files in {root}.")
    if sample_count is not None:
        if len(pairs) < sample_count:
            raise ValueError(f"Requested {sample_count} files but only {len(pairs)} match pairs exist.")
        pairs = random.Random(seed).sample(pairs, sample_count)
    return pairs


def sample_disjoint_train_val_pairs(
    train_dir: str,
    val_dir: str,
    train_file_count: int | None,
    val_file_count: int | None,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Sample validation first, then exclude its match IDs from training.

    This also protects against accidental overlap when different directories
    contain copied versions of the same StatsBomb match.
    """
    train_population = find_match_pairs(train_dir, sample_count=None, seed=seed)
    val_population = find_match_pairs(val_dir, sample_count=None, seed=seed + 1)

    if val_file_count is not None:
        if len(val_population) < val_file_count:
            raise ValueError(
                f"Requested {val_file_count} validation files but only "
                f"{len(val_population)} match pairs exist."
            )
        val_pairs = random.Random(seed + 1).sample(val_population, val_file_count)
    else:
        val_pairs = val_population

    validation_match_ids = {match_id_from_filename(Path(pair[0])) for pair in val_pairs}
    disjoint_train_population = [
        pair
        for pair in train_population
        if match_id_from_filename(Path(pair[0])) not in validation_match_ids
    ]
    if train_file_count is not None:
        if len(disjoint_train_population) < train_file_count:
            raise ValueError(
                f"Requested {train_file_count} training files, but only "
                f"{len(disjoint_train_population)} remain after excluding "
                f"{len(validation_match_ids)} validation match IDs."
            )
        train_pairs = random.Random(seed).sample(disjoint_train_population, train_file_count)
    else:
        train_pairs = disjoint_train_population

    return train_pairs, val_pairs


def load_match_datasets(
    pairs: Sequence[tuple[str, str]],
    factory: Callable[[str, str], Dataset],
) -> tuple[Dataset, list[tuple[str, str]]]:
    """Build one dataset per match, skipping (and returning) pairs that yield no samples.

    StatsBomb open-data has broken matches upstream: one corrupt 360 file
    (3845506 contains a block of null bytes), seven whose 360 frames link to
    none of the events, and one with a single frame.  None of them may abort a
    multi-hour run or silently inflate the match count.
    """
    datasets = []
    skipped: list[tuple[str, str]] = []
    for freeze_frame_path, event_path in pairs:
        try:
            dataset = factory(freeze_frame_path, event_path)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            print(f"WARNING: skipping unreadable match ({freeze_frame_path}, {event_path}): {error}")
            skipped.append((freeze_frame_path, event_path))
            continue
        if len(dataset) == 0:
            print(f"WARNING: skipping match without usable 360 samples ({freeze_frame_path}, {event_path})")
            skipped.append((freeze_frame_path, event_path))
            continue
        datasets.append(dataset)
    if not datasets:
        raise ValueError("No readable matches were loaded.")
    return ConcatDataset(datasets), skipped


def sample_fingerprint(dataset: Dataset) -> str:
    """Hash target event IDs, making a benchmark split auditable and repeatable."""
    identifiers: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = getattr(sample, "metadata", None) or {}
        identifiers.append(str(metadata.get("event_uuid", index)))
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()[:16]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_fold_pairs(
    data_dir: str, num_folds: int, fold: int, seed: int, monitor_matches: int | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """K-fold split of all matches for cross-fitting: (train, monitoring validation, full fold).

    Matches are shuffled deterministically with ``seed`` and cut into
    ``num_folds`` contiguous folds; fold ``fold`` is held out and every other
    match trains.  Over all folds every match is held out exactly once, so each
    gets out-of-sample predictions.  ``monitor_matches`` limits the per-epoch
    validation to the first matches of the held-out fold (checkpoint selection
    only; the full fold is returned for later out-of-fold predictions).
    """
    if not 0 <= fold < num_folds:
        raise ValueError(f"fold must lie in [0, {num_folds}).")
    pairs = find_match_pairs(data_dir, sample_count=None, seed=seed)
    random.Random(seed).shuffle(pairs)
    bounds = [round(i * len(pairs) / num_folds) for i in range(num_folds + 1)]
    held_out = pairs[bounds[fold] : bounds[fold + 1]]
    train = pairs[: bounds[fold]] + pairs[bounds[fold + 1] :]
    monitor = held_out[:monitor_matches] if monitor_matches else held_out
    return train, monitor, held_out
