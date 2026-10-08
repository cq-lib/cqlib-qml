# cqlib_qml/algorithms/HQNN_classification.py
"""
Hybrid Quantum Neural Network for classification tasks.

This module provides a complete training pipeline for Hybrid Quantum Neural Networks
(HQNNs) on classification problems. It includes training and validation loops with
checkpoint saving, TensorBoard logging, and configuration management.

The HQNN combines a parameterized quantum circuit (ansatz) with a classical
fully-connected layer, leveraging the strengths of both quantum and classical
computing paradigms.

Examples:
    >>> from cqlib_qml.algorithms.HQNN_classification import train, validate
    >>> from cqlib_qml.models import HQNN
    >>> from cqlib_qml.ansatz import HEAnsatz
    >>> from cqlib_qml.encoder import FRQI
    >>>
    >>> # Setup model
    >>> ansatz = HEAnsatz(n_qubits=5, d=3, layers=["RY", "CX"])
    >>> model = HQNN(ansatz=ansatz, out_dim=3)
    >>>
    >>> # Train
    >>> for epoch in range(EPOCHS):
    ...     train(epoch, 0, model, train_loader, loss_fun, model_path, EPOCHS, tb)
    ...     validate(epoch, model, test_loader, loss_fun, BATCH_SIZE, tb)
"""

import time
from pathlib import Path
import tqdm
import yaml
import numpy as np
from typing import Optional, Any

from cqlib_qml.data.data_preprocess import get_mnist_dataloader
from cqlib_qml.models import HQNN
from cqlib_qml.ansatz import *
from cqlib_qml.encoder import *
from cqlib_qml.loss import *
from cqlib_qml.optimizer import *

try:
    import torch.utils.tensorboard

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def train(
    ep: int,
    it_start: int,
    net: HQNN,
    train_loader,
    loss_fun,
    model_path: str,
    total_epochs: int,
    tb: Optional[Any] = None,
) -> None:
    """
    Train the HQNN model for one epoch.

    This function iterates through the training data, computes forward and
    backward passes, updates model parameters, and logs training metrics.

    Args:
        ep (int): Current epoch number (0-indexed).
        it_start (int): Starting iteration index within the epoch. Used for
            resuming training from a checkpoint.
        net (HQNN): The Hybrid Quantum Neural Network model to train.
        train_loader: DataLoader for training data.
        loss_fun: Loss function instance.
        model_path (str): Path to save checkpoints.
        total_epochs (int): Total number of epochs for training.
        tb (optional): TensorBoard SummaryWriter instance for logging.
            If None, logging is disabled. Defaults to None.

    Note:
        The function uses the global variable `EPOCH` defined in the main script
        for determining the latest checkpoint.

    Examples:
        >>> for epoch in range(10):
        ...     train(epoch, 0, model, train_loader, loss_fun, "./checkpoints/", 10, tensorboard_writer)
    """
    restored = getattr(net, "_resume_data_loader_state", None)
    enumeration_start = 0
    if restored is not None:
        if not hasattr(train_loader, "load_state_dict"):
            raise ValueError("Exact resume requires the project DataLoader")
        train_loader.load_state_dict(restored)
        enumeration_start = 0 if restored["next_batch"] == len(train_loader) else restored["next_batch"]
        net._resume_data_loader_state = None
        it_start = enumeration_start
    loader = tqdm.tqdm(train_loader, desc="Training epoch {}".format(ep + 1), leave=True)
    for it, (x_train, y_train) in enumerate(loader, start=enumeration_start):
        if it < it_start:
            continue
        it_start = 0
        expectations = net.forward(x_train)  # (batch_size, n_classes)
        # Prepare labels for CrossEntropy loss
        y_true = np.zeros(expectations.shape)
        for i in range(len(y_train)):
            idx = y_train[i]
            y_true[i, idx] = 1
        y_pred = expectations

        # Compute loss
        loss = loss_fun(y_pred, y_true)
        # Optimize: backward pass and parameter update
        net.backward(loss_fun.grads())
        net.update(cur_loss=loss)
        net.zero_grad()

        # Compute accuracy
        correct = sum(y_pred.argmax(axis=1) == y_train)
        accuracy = correct / len(y_train)
        loader.set_postfix(it=it, loss="{:.3f}".format(loss), accuracy="{:.3f}".format(accuracy))
        # Log metrics
        if tb is not None:
            tb.add_scalar("train/loss", loss, ep * len(loader) + it)
            tb.add_scalar("train/accuracy", accuracy, ep * len(loader) + it)
        # Save checkpoint
        latest = (ep + 1) == total_epochs and (it + 1) == len(loader)
        if (it != 0 and it % 30 == 0) or latest:
            net.save_checkpoint(model_path, ep, it, latest, data_loader=train_loader if hasattr(train_loader, "state_dict") else None)


def validate(ep: int, net: HQNN, test_loader, loss_fun, batch_size: int, tb: Optional[Any] = None) -> tuple:
    """
    Validate the HQNN model on the test dataset.

    This function evaluates the model's performance on the test set and
    computes average loss and accuracy across all batches.

    Args:
        ep (int): Current epoch number (0-indexed).
        net (HQNN): The Hybrid Quantum Neural Network model to validate.
        test_loader: DataLoader for test data.
        loss_fun: Loss function instance.
        batch_size (int): Batch size used in the dataloader.
        tb (optional): TensorBoard SummaryWriter instance for logging.
            If None, logging is disabled. Defaults to None.

    Returns:
        tuple: (avg_loss, avg_acc) where:
            - avg_loss (float): Average loss over the validation set.
            - avg_acc (float): Average accuracy over the validation set.

    Note:
        Model is evaluated in non-trainable mode.

    Examples:
        >>> avg_loss, avg_acc = validate(epoch, model, test_loader, loss_fun, 32, tensorboard_writer)
        >>> print(f"Validation: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    """
    loader_val = tqdm.tqdm(test_loader, desc="Validating epoch {}".format(ep + 1), leave=True)
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    for it, (x_test, y_test) in enumerate(loader_val):
        expectations_val = net.forward(x_test, trainable=False)
        # Prepare labels for CrossEntropy loss
        y_true_val = np.zeros(expectations_val.shape)
        for i in range(len(y_test)):
            idx = y_test[i]
            y_true_val[i, idx] = 1
        y_pred_val = expectations_val

        loss_val = loss_fun(y_pred_val, y_true_val)
        count = len(y_test)
        total_samples += count
        total_loss += loss_val if getattr(loss_fun, "reduction", "mean") == "sum" else loss_val * count

        # Compute accuracy
        correct_val = sum(y_pred_val.argmax(axis=1) == y_test)
        total_correct += correct_val
        accuracy_val = correct_val / len(y_test)
        loader_val.set_postfix(
            it=it,
            loss="{:.3f}".format(loss_val),
            accuracy="{:.3f}".format(accuracy_val),
        )

    if not total_samples:
        raise ValueError("Validation dataset must not be empty")
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    if tb is not None:
        tb.add_scalar("validation/loss", avg_loss, ep)
        tb.add_scalar("validation/accuracy", avg_acc, ep)
    print("Validation Average Loss: {}, Accuracy: {}".format(avg_loss, avg_acc))
    return avg_loss, avg_acc


if __name__ == "__main__":
    """
    Main execution block for training HQNN on MNIST dataset.

    This script demonstrates a complete training pipeline including:
        1. Data loading and preprocessing (MNIST with FRQI encoding)
        2. Model initialization (HEAnsatz + HQNN)
        3. Training loop with progress tracking
        4. Validation and checkpoint saving
        5. Configuration logging

    Configuration:
        - Dataset: MNIST (classes 0, 1, 2)
        - Image size: 4x4 pixels
        - Grayscale: 2 levels
        - Encoding: FRQI (Flexible Representation of Quantum Images)
        - Ansatz: HEAnsatz (Hardware-Efficient)
        - Loss: SoftmaxCrossEntropy
        - Optimizer: Adam (lr=0.01)
        - Epochs: 10
        - Batch size: 32
    """
    # Training parameters
    EPOCH = 10
    BATCH_SIZE = 32
    LR = 0.01
    SEED = 0
    ep_start = 0
    it_start = 0
    np.random.seed(SEED)

    classes = [0, 1, 2]
    RESIZE = (4, 4)
    GRAYSCALE = 2
    n_pixels = RESIZE[0] * RESIZE[1]
    # Initialize encoder, loss function, and optimizer
    encoding = FRQI(n_pixels, GRAYSCALE)
    loss_fun = SoftmaxCrossEntropy()
    optimizer = Adam(lr=LR)

    # Setup quantum circuit parameters
    pos_qubits = int(np.log2(n_pixels))
    color_qubits = 1
    n_qubits = pos_qubits + color_qubits

    # Initialize ansatz and model
    ansatz = HEAnsatz(n_qubits, d=5, layers=["RZ", "RY", "RZ", "CX"])
    train_loader, test_loader = get_mnist_dataloader(classes, RESIZE, encoding, BATCH_SIZE, GRAYSCALE)
    net = HQNN(ansatz=ansatz, out_dim=len(classes), optimizer=optimizer)

    # Setup logging and checkpoint directories
    now_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    model_path = "./HQNN_MNIST_" + now_time + "/"
    Path(model_path).mkdir(parents=True, exist_ok=True)
    tb = torch.utils.tensorboard.SummaryWriter(log_dir=model_path + "logs") if TENSORBOARD_AVAILABLE else None

    # Save configuration
    config = {
        "classes": str(classes),
        "encoding": str(encoding),
        "ansatz": str(ansatz),
        "grayscale": GRAYSCALE,
        "resize": str(RESIZE),
        "LR": LR,
        "loss_fun": str(loss_fun),
        "seed": SEED,
    }
    with open(model_path + "config.yaml", "w") as config_file:
        config_file.write(yaml.dump(config))

    # Training loop
    for ep in range(ep_start, EPOCH):
        if ep > ep_start:
            it_start = 0
        train(ep, it_start, net, train_loader, loss_fun, model_path, EPOCH, tb)
        avg_loss, avg_acc = validate(ep, net, test_loader, loss_fun, BATCH_SIZE, tb)
        with open(model_path + "result.yaml", "a") as result_file:
            result_file.write("Validation Average Loss: {}, Accuracy: {}\n".format(avg_loss, avg_acc))
