import torch
from homework.dataset import CIFAR10Dataset
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader


def eval_clip(
    dataset: CIFAR10Dataset,
    classes: list[str],
    clip_model,
    tokenizer,
    batch_size: int,
    shuffle: bool = False,
    print_classification_report: bool = True,
) -> list[int]:
    logger.info("Tokenizing classes")
    tokenized_classes = tokenizer(classes).to("cuda")
    y_true = []
    y_pred = []

    logger.info("Generating DataLoader")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    logger.info("Starting evaluation")
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda"), y.to("cuda")

        logger.info(f"\tProcessing batch {batch_idx}")
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = clip_model.encode_image(X)
            text_features = clip_model.encode_text(tokenized_classes)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            pred = text_probs.argmax(dim=-1)

            logger.info(f"\tBatch accuracy: {accuracy_score(y.cpu(), pred.cpu())}")

            y_pred += pred.cpu().tolist()
            y_true += y.cpu().tolist()

    if print_classification_report:
        logger.info("Classification Report:")
        logger.info(
            "\n"
            + classification_report(
                y_true, y_pred, labels=range(len(classes)), target_names=classes
            )
        )

    return y_true, y_pred
