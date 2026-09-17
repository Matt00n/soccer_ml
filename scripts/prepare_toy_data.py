"""Utility to prepare toy receiver prediction dataset from open-data."""

import argparse
import json
from pathlib import Path
from typing import List, Tuple


def find_match_with_360_data(
    open_data_dir: Path,
    num_matches: int = 3,
) -> List[Tuple[int, Path, Path]]:
    """Find matches that have both events and 360 data.

    Args:
        open_data_dir: Path to data/open-data directory
        num_matches: Number of matches to return

    Returns:
        List of (match_id, events_path, three_sixty_path) tuples
    """
    events_dir = open_data_dir / "data" / "events"
    three_sixty_dir = open_data_dir / "data" / "three-sixty"

    matches = []

    # Iterate through 360 files
    for three_sixty_file in sorted(three_sixty_dir.glob("*.json"))[
        :100
    ]:  # Limit to first 100
        match_id_str = three_sixty_file.stem

        # Try to parse match ID
        try:
            match_id = int(match_id_str)
        except ValueError:
            continue

        # Check if corresponding events file exists
        events_file = events_dir / f"{match_id_str}.json"
        if not events_file.exists():
            continue

        matches.append((match_id, events_file, three_sixty_file))

        if len(matches) >= num_matches:
            break

    return matches


def copy_toy_data(
    matches: List[Tuple[int, Path, Path]],
    output_dir: Path,
):
    """Copy selected match data to output directory.

    Args:
        matches: List of (match_id, events_path, three_sixty_path) tuples
        output_dir: Directory to save toy data
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for match_id, events_file, three_sixty_file in matches:
        # Copy events
        output_events = output_dir / f"events_{match_id}.json"
        with open(events_file, "r") as src, open(output_events, "w") as dst:
            json.dump(json.load(src), dst)
        print(f"Copied events: {output_events}")

        # Copy 360 data
        output_360 = output_dir / f"three_sixty_{match_id}.json"
        with open(three_sixty_file, "r") as src, open(output_360, "w") as dst:
            json.dump(json.load(src), dst)
        print(f"Copied 360: {output_360}")

    print(f"\nToy data saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare toy receiver prediction dataset"
    )
    parser.add_argument(
        "--open-data",
        type=str,
        default="data/open-data",
        help="Path to data/open-data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_toy_receiver",
        help="Output directory for toy data",
    )
    parser.add_argument(
        "--num-matches",
        type=int,
        default=3,
        help="Number of matches to extract",
    )

    args = parser.parse_args()

    open_data_dir = Path(args.open_data)
    output_dir = Path(args.output)

    if not open_data_dir.exists():
        print(f"Error: {open_data_dir} not found")
        return

    # Find matches
    print(f"Finding matches with 360 data in {open_data_dir}...")
    matches = find_match_with_360_data(open_data_dir, num_matches=args.num_matches)

    if not matches:
        print("No matches with 360 data found!")
        return

    print(f"Found {len(matches)} matches:")
    for match_id, _, _ in matches:
        print(f"  - Match {match_id}")

    # Copy toy data
    print(f"\nCopying to {output_dir}...")
    copy_toy_data(matches, output_dir)


if __name__ == "__main__":
    main()
