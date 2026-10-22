"""Find xBD scenes on disk and pair each pre/post image with its label files.

Expected layout (verified against the real download by scripts/inspect_dataset.py):

    <dataset_root>/<xbd_split>/images/<event>_<8 digits>_<pre|post>_disaster.png
    <dataset_root>/<xbd_split>/labels/<event>_<8 digits>_<pre|post>_disaster.json

"xbd_split" is xBD's own folder (train / tier3 / test / hold). It is NOT AfterSight's
train/val/test split, which is made later and grouped by scene.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FILENAME_STEM = re.compile(r"^(?P<event>[a-z0-9]+(?:-[a-z0-9]+)*)_(?P<number>\d{8})_(?P<phase>pre|post)_disaster$")


@dataclass
class SceneFiles:
    scene_id: str  # "<event>_<number>", e.g. "hurricane-michael_00000123"
    event: str
    xbd_split: str
    pre_image: Path | None = None
    post_image: Path | None = None
    pre_label: Path | None = None
    post_label: Path | None = None

    @property
    def missing_parts(self) -> list[str]:
        parts = {
            "pre_image": self.pre_image,
            "post_image": self.post_image,
            "pre_label": self.pre_label,
            "post_label": self.post_label,
        }
        return [name for name, path in parts.items() if path is None]

    @property
    def complete(self) -> bool:
        return not self.missing_parts

    def files(self) -> list[Path]:
        return [p for p in (self.pre_image, self.post_image, self.pre_label, self.post_label) if p is not None]


@dataclass
class DiscoveryResult:
    root_exists: bool
    root_subdirs: list[str]  # shown to the user when the layout is not what we expect
    splits_found: list[str]
    scenes: list[SceneFiles]  # configured events only, sorted by (xbd_split, scene_id)
    other_events: dict[str, int]  # scenes present on disk for events NOT in the config
    unexpected_files: list[str] = field(default_factory=list)  # relative paths, capped
    unexpected_file_count: int = 0
    duplicate_scene_ids: list[str] = field(default_factory=list)  # same scene in two xbd splits


MAX_UNEXPECTED_LISTED = 50


def discover_scenes(root: Path, xbd_splits: list[str], events: list[str], image_extensions: list[str]) -> DiscoveryResult:
    if not root.is_dir():
        return DiscoveryResult(False, [], [], [], {})

    root_subdirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    wanted = set(events)
    scenes: dict[tuple[str, str], SceneFiles] = {}
    other_events: Counter[str] = Counter()
    other_seen: set[tuple[str, str]] = set()
    unexpected: list[str] = []
    unexpected_count = 0
    splits_found: list[str] = []

    def note_unexpected(path: Path) -> None:
        nonlocal unexpected_count
        unexpected_count += 1
        if len(unexpected) < MAX_UNEXPECTED_LISTED:
            unexpected.append(str(path.relative_to(root)))

    for split in xbd_splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        splits_found.append(split)
        for kind, allowed_ext in (("images", set(image_extensions)), ("labels", {".json"})):
            folder = split_dir / kind
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                match = FILENAME_STEM.match(path.stem)
                if not path.is_file() or path.suffix.lower() not in allowed_ext or not match:
                    note_unexpected(path)
                    continue
                event, phase = match["event"], match["phase"]
                scene_id = f"{event}_{match['number']}"
                if event not in wanted:
                    if (split, scene_id) not in other_seen:
                        other_seen.add((split, scene_id))
                        other_events[event] += 1
                    continue
                scene = scenes.setdefault((split, scene_id), SceneFiles(scene_id, event, split))
                attr = f"{phase}_{'image' if kind == 'images' else 'label'}"
                if getattr(scene, attr) is not None:
                    note_unexpected(path)  # e.g. both .png and .tif for the same scene
                    continue
                setattr(scene, attr, path)

    id_counts = Counter(scene_id for _split, scene_id in scenes)
    duplicates = sorted(sid for sid, n in id_counts.items() if n > 1)

    return DiscoveryResult(
        root_exists=True,
        root_subdirs=root_subdirs,
        splits_found=splits_found,
        scenes=[scenes[key] for key in sorted(scenes)],
        other_events=dict(sorted(other_events.items())),
        unexpected_files=unexpected,
        unexpected_file_count=unexpected_count,
        duplicate_scene_ids=duplicates,
    )
