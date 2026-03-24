"""GID dataset reader for unified JSCC benchmark.

Expected layout:
- images/<video_name>/*.(jpg|png)
- aligned_depths/<video_name>/*.png
- (optional) instance-masks/<video_name>/*-instance.png or <frame>.png
- (optional) box2d/<video_name>.json
- video-train.txt, video-test.txt
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class GIDSample:
    video_name: str
    frame_stem: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None
    frame_box2d: List[List[float]] | None


class GIDFramePairDataset(Dataset):
    """Frame-paired RGB/depth loader for GID.

    Design choice for current benchmark scaffolding:
    - RGB + aligned depth are required.
    - instance masks are optional metadata (`use_instance_masks=False` by default).
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "test"],
        image_size: tuple[int, int] = (224, 224),
        depth_scale_m: float = 10.0,
        use_instance_masks: bool = False,
        use_box2d: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.depth_scale_m = depth_scale_m
        self.use_instance_masks = use_instance_masks
        self.use_box2d = use_box2d
        self.box2d_root = self.root / "box2d"

        self.rgb_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.depth_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self._box2d_cache = self._load_box2d_cache()
        self.samples = self._index_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found for split={split} under {root}")

    def _load_split_videos(self) -> List[str]:
        split_file = self.root / f"video-{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _index_samples(self) -> List[GIDSample]:
        videos = self._load_split_videos()
        samples: List[GIDSample] = []

        for video_name in videos:
            rgb_dir = self.root / "images" / video_name
            depth_dir = self.root / "aligned_depths" / video_name
            mask_dir = self.root / "instance-masks" / video_name

            if not rgb_dir.exists() or not depth_dir.exists():
                continue

            rgb_paths = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
            for rgb_path in sorted(rgb_paths):
                stem = rgb_path.stem
                depth_path = depth_dir / f"{stem}.png"
                if not depth_path.exists():
                    continue
                mask_path = mask_dir / f"{stem}.png"
                if not mask_path.exists():
                    mask_path = mask_dir / f"{stem}-instance.png"
                frame_box2d = None
                if self.use_box2d:
                    frame_box2d = self._box2d_cache.get(video_name, {}).get(stem)
                samples.append(
                    GIDSample(
                        video_name=video_name,
                        frame_stem=stem,
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        mask_path=mask_path if mask_path.exists() else None,
                        frame_box2d=frame_box2d,
                    )
                )
        return samples

    def _load_box2d_cache(self) -> Dict[str, Dict[str, List[List[float]]]]:
        if not self.use_box2d:
            return {}
        cache: Dict[str, Dict[str, List[List[float]]]] = {}
        if not self.box2d_root.exists():
            return cache
        for p in sorted(self.box2d_root.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, dict):
                cache[p.stem] = data
        return cache

    def __len__(self) -> int:
        return len(self.samples)

    def _load_depth_tensor(self, path: Path) -> torch.Tensor:
        depth_img = Image.open(path).convert("L")
        depth = self.depth_tf(depth_img)  # [1, H, W], 0..1
        # align to metric-like range as documented by GID (0.01m..10.0m)
        depth = torch.clamp(depth * self.depth_scale_m, min=0.01, max=self.depth_scale_m)
        return depth

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        s = self.samples[index]
        rgb = self.rgb_tf(Image.open(s.rgb_path).convert("RGB"))
        depth = self._load_depth_tensor(s.depth_path)

        out: Dict[str, torch.Tensor | str] = {
            "video_name": s.video_name,
            "frame_id": s.frame_stem,
            "rgb": rgb,
            "depth": depth,
        }

        if self.use_instance_masks:
            if s.mask_path is None:
                mask = torch.zeros((1, rgb.shape[-2], rgb.shape[-1]), dtype=torch.int64)
            else:
                mask_raw = Image.open(s.mask_path)
                mask_np = np.array(mask_raw, dtype=np.int64)
                mask = torch.from_numpy(mask_np).unsqueeze(0)
            out["instance_mask"] = mask

        if self.use_box2d:
            boxes = s.frame_box2d if s.frame_box2d is not None else []
            out["box2d"] = torch.tensor(boxes, dtype=torch.float32)

        return out
