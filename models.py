import json
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from timm import create_model
from ultralytics import YOLO
from pathlib import Path
from config import DEVICE

def load_yolo_model(weights_path: Path):
    print(f"Loading YOLOv8 from {weights_path}")
    return YOLO(str(weights_path))


def _load_json_meta_if_exists(meta_path: Path):
    if meta_path is not None and meta_path.exists():
        with open(meta_path, "r") as f:
            return json.load(f)
    return None


def load_efficientnet_classifier(weights_path: Path,
                                 meta_json_path: Path | None = None):
    """
    Generic EfficientNet-based image classifier loader (for SHOT and RUNOUT).
    """
    print(f"Loading EfficientNet classifier from {weights_path}")

    model_name = "efficientnet_b0"
    class_names = None
    meta = _load_json_meta_if_exists(meta_json_path)
    if meta is not None:
        class_names = meta.get("class_names")
        model_name = meta.get("model_name", model_name)

    ckpt = torch.load(str(weights_path), map_location=DEVICE)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        if class_names is None and "class_names" in ckpt:
            class_names = ckpt["class_names"]
        if "model_name" in ckpt:
            model_name = ckpt["model_name"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if class_names is None and "class_names" in ckpt:
            class_names = ckpt["class_names"]
        if "model_name" in ckpt:
            model_name = ckpt["model_name"]
    else:
        state_dict = ckpt

    if class_names is None:
        raise RuntimeError(
            f"Could not find class_names for {weights_path}. "
            f"Please provide a JSON with 'class_names', or include it in the checkpoint."
        )

    num_classes = len(class_names)

    print(f"  -> model_name={model_name}, num_classes={num_classes}")
    print(f"  -> class_names={class_names}")

    base_model_name = model_name
    if model_name.startswith("efficientnet_b0") and model_name != "efficientnet_b0":
        print(f"  -> Detected custom variant '{model_name}', using backbone 'efficientnet_b0'")
        base_model_name = "efficientnet_b0"

    model = create_model(base_model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, class_names


class UmpireEfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.base = efficientnet_b0(weights=weights)
        in_feat = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.base(x)


def load_umpire_model(weights_path: Path,
                      meta_json_path: Path | None = None):
    print(f"Loading Umpire EfficientNet model from {weights_path}")

    class_names = None
    meta = _load_json_meta_if_exists(meta_json_path)
    if meta is not None:
        class_names = meta.get("class_names")

    ckpt = torch.load(str(weights_path), map_location=DEVICE)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        if class_names is None and "class_names" in ckpt:
            class_names = ckpt["class_names"]
    else:
        state_dict = ckpt

    if class_names is None:
        raise RuntimeError(
            f"[UMPIRE] Could not find class_names for {weights_path}. "
            f"Ensure the checkpoint or meta JSON has 'class_names'."
        )

    num_classes = len(class_names)
    print(f"  -> num_classes={num_classes}")
    print(f"  -> class_names={class_names}")

    model = UmpireEfficientNetClassifier(num_classes).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names


def load_r2plus1d_model(weights_path: Path,
                        meta_json_path: Path | None = None):
    print(f"Loading R(2+1)D model from {weights_path}")

    class_names = None
    meta = _load_json_meta_if_exists(meta_json_path)
    if meta is not None:
        class_names = meta.get("class_names")

    ckpt = torch.load(str(weights_path), map_location=DEVICE)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        if class_names is None and "class_names" in ckpt:
            class_names = ckpt["class_names"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if class_names is None and "class_names" in ckpt:
            class_names = ckpt["class_names"]
    else:
        state_dict = ckpt

    if class_names is None:
        raise RuntimeError(
            f"Could not find class_names for {weights_path}. "
            f"Please provide a JSON with 'class_names', or include it in the checkpoint."
        )

    num_classes = len(class_names)
    print(f"  -> num_classes={num_classes}")
    print(f"  -> class_names={class_names}")

    model = r2plus1d_18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, class_names
