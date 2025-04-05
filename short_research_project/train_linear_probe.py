from typing import Literal

import torch
from homework.dataset import CIFAR10Dataset
from loguru import logger
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader


def train_linear_probe_model(
    linear_probe_model: torch.nn.Module,
    train_dataset: CIFAR10Dataset,
    eval_dataset: CIFAR10Dataset,
    clip_model,
    device: Literal["cpu", "cuda"],
    batch_size: int,
    learning_rate: float,
    epochs: int,
    log_every_n_steps: int,
):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_probe_model.parameters(), lr=learning_rate)

    # set clip model in eval mode.
    clip_model.eval()
    for epoch in range(epochs):
        # -------------------------------------------------
        # Train

        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            # calculate image features with the clip model frozen.
            with torch.no_grad(), torch.autocast("cuda"):
                image_features = clip_model.encode_image(X)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            # linear probe model optimization step
            linear_probe_model.train()
            with torch.autocast("cuda"):
                logits = linear_probe_model(image_features)
                loss = loss_fn(logits, y)

                if batch_idx % log_every_n_steps == 0:
                    logger.info(
                        f"[Training] Epoch {epoch + 1} | Batch {batch_idx + 1} | "
                        f"Loss = {round(loss.item(), 4)}"
                    )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # -------------------------------------------------
        # Evaluate
        logger.info("Evaluating in test_dataset")

        y_true = []
        y_pred = []
        linear_probe_model.eval()

        for batch_idx, (X, y) in enumerate(eval_dataloader):
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
                        f"[Test] Epoch {epoch + 1} | Batch {batch_idx + 1} | "
                        "accuracy = "
                        f"{accuracy_score(y.cpu().tolist(), pred.cpu().tolist())}"
                    )

        logger.info(
            f"[Test] Epoch {epoch + 1} accuracy: {accuracy_score(y_true, y_pred)}"
        )
