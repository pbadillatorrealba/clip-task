from typing import Literal

import open_clip


def load_clip(device: Literal["cpu", "cuda"]) -> tuple:
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    return model, preprocess, tokenizer
