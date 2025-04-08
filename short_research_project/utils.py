import pickle

import pandas as pd
import plotly.express as px
from sklearn.metrics import classification_report


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


def get_scores(y_true, y_pred):
    clf_report = classification_report(y_true, y_pred, output_dict=True)

    recall = [
        [class_, values["recall"]]
        for class_, values in clf_report.items()
        if class_ != "accuracy" and class_ != "macro avg" and class_ != "weighted avg"
    ]
    accuracy = [["accuracy", clf_report["accuracy"]]]

    return recall + accuracy


def get_evaluation_dataframe(
    y_true_clip,
    y_pred_clip,
    y_true_linear_probe,
    y_pred_linear_probe,
    y_true_prompted,
    y_pred_prompted,
    y_true_prompted_2,
    y_pred_prompted_2,
):
    scores_clip = pd.DataFrame(
        get_scores(y_true_clip, y_pred_clip), columns=["class", "scores_clip"]
    )
    scores_linear_probe = pd.DataFrame(
        get_scores(y_true_linear_probe, y_pred_linear_probe),
        columns=["class", "scores_linear_probe"],
    )
    scores_prompted = pd.DataFrame(
        get_scores(y_true_prompted, y_pred_prompted),
        columns=["class", "scores_prompted"],
    )
    scores_prompted_2 = pd.DataFrame(
        get_scores(y_true_prompted_2, y_pred_prompted_2),
        columns=["class", "scores_prompted_2"],
    )

    df = (
        pd.merge(
            pd.merge(
                pd.merge(scores_clip, scores_linear_probe, on="class"),
                scores_prompted,
                on="class",
            ),
            scores_prompted_2,
            on="class",
        )
        .replace(
            {
                "class": {
                    "0": "airplane",
                    "1": "automobile",
                    "2": "bird",
                    "3": "cat",
                    "4": "deer",
                    "5": "dog",
                    "6": "frog",
                    "7": "horse",
                    "8": "ship",
                    "9": "truck",
                }
            }
        )
        .rename(
            columns={
                "class": "Class",
                "scores_clip": "Zero-Shot",
                "scores_linear_probe": "Linear Probe",
                "scores_prompted": "Prompting (Template 1)",
                "scores_prompted_2": "Prompting (Template 2)",
            }
        )
        .set_index("Class")
        .T.round(2)
    )

    return df
