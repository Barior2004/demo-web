import argparse
import hashlib
from pathlib import Path

import torch
from ultralytics import YOLO


class CoordAttn(torch.nn.Module):
    """Coordinate Attention module required to load the custom checkpoint."""

    def forward(self, x):
        identity = x
        _, _, height, width = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(
            y,
            [height, width],
            dim=2,
        )

        x_w = x_w.permute(0, 1, 3, 2)

        attention_h = self.conv_h(x_h).sigmoid()
        attention_w = self.conv_w(x_w).sigmoid()

        return identity * attention_h * attention_w


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export YOLOv10 CoordAttn checkpoint "
            "from PyTorch to ONNX."
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Đường dẫn đến file checkpoint best.pt.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Đường dẫn file ONNX đầu ra.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    source = args.weights.expanduser().resolve()
    target = args.output.expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {source}"
        )

    if source.suffix.lower() != ".pt":
        raise ValueError(
            "Checkpoint đầu vào phải có phần mở rộng .pt"
        )

    if target.suffix.lower() != ".onnx":
        raise ValueError(
            "File đầu ra phải có phần mở rộng .onnx"
        )

    if target.exists():
        raise FileExistsError(
            f"File đầu ra đã tồn tại: {target}"
        )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(
        str(source),
        task="detect",
    )

    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            batch=1,
            dynamic=False,
            simplify=True,
            opset=18,
            device="cpu",
        )
    ).resolve()

    if exported != target:
        exported.replace(target)

    sha256 = calculate_sha256(target)

    print(f"Đã export: {target}")
    print(f"SHA256: {sha256}")


if __name__ == "__main__":
    main()