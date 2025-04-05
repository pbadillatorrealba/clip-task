import pickle

import plotly.express as px


def plot_example(dataset: dict, index: int, meta: dict):
    img, label = dataset[index]
    px.imshow(
        img.permute(1, 2, 0) * 128,
        title=f"Class {label} - {meta[b'label_names'][label]}",
        height=400,
    ).show()


def load_meta(path: str) -> dict:
    with open(path, "rb") as fo:
        meta = pickle.load(fo, encoding="bytes")
    return meta
