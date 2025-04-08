import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from short_research_project.dataset import CIFAR10Dataset


def eval_clip(
    dataset: CIFAR10Dataset,
    classes: list[str],
    clip_model,
    tokenizer,
    batch_size: int,
    shuffle: bool = False,
    print_classification_report: bool = True,
) -> tuple[list[int], list[int]]:
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


def eval_linear_probe_model(
    linear_probe_model: torch.nn.Module,
    test_dataset: CIFAR10Dataset,
    classes: list[str],
    clip_model,
    batch_size: int,
    shuffle: bool = False,
    print_classification_report: bool = True,
) -> tuple[list[int], list[int]]:
    logger.info("Tokenizing classes")
    y_true = []
    y_pred = []

    logger.info("Generating DataLoader")
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    logger.info("Starting evaluation")
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to("cuda"), y.to("cuda")

        logger.info(f"\tProcessing batch {batch_idx}")
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = clip_model.encode_image(X)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = linear_probe_model(image_features)
            pred = logits.argmax(dim=-1)

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


def evalute_linear_probe_model(
    test_dataset,
    linear_probe_model,
    clip_model,
    batch_size,
    device="cuda",
    log_every_n_steps=5,
):
    # -------------------------------------------------
    # Evaluate
    logger.info("Evaluating in test_dataset")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    y_true = []
    y_pred = []
    linear_probe_model.eval()

    for batch_idx, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        with torch.no_grad(), torch.autocast("cuda"):
            # calculate image featues.
            image_features = clip_model.encode_image(X)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.float()

            # calculate class using linear probe model.
            logits = linear_probe_model(image_features)
            pred = logits.argmax(dim=-1)

            y_pred += pred.cpu().tolist()
            y_true += y.cpu().tolist()
            # report

            if batch_idx % log_every_n_steps == 0:
                logger.info(
                    f"[Test] Batch {batch_idx + 1} | "
                    "accuracy = "
                    f"{accuracy_score(y.cpu().tolist(), pred.cpu().tolist())}"
                )

    logger.info(f"[Test] accuracy: {accuracy_score(y_true, y_pred)}")

    return y_true, y_pred
