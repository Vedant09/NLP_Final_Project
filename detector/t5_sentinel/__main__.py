import os
import random
import wandb
import torch
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import torch.cuda as cuda
import torch.optim as optim
from torch.utils.data import DataLoader
from detector.t5_sentinel.dataset import Dataset
from detector.t5_sentinel.model import Sentinel
from detector.t5_sentinel.utilities import train, validate
from detector.t5_sentinel.__init__ import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

##############################################################################
# Dataset and Dataloader
##############################################################################
train_loader = DataLoader(
    train_dataset := Dataset("train-dirty"),
    collate_fn=train_dataset.collate_fn,
    batch_size=config.dataloader.batch_size,
    num_workers=config.dataloader.num_workers,
    shuffle=True,
)


valid_loader = DataLoader(
    valid_dataset := Dataset("valid-dirty"),
    collate_fn=valid_dataset.collate_fn,
    batch_size=config.dataloader.batch_size,
    num_workers=config.dataloader.num_workers,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset := Dataset("test-dirty"),
    collate_fn=test_dataset.collate_fn,
    batch_size=config.dataloader.batch_size,
    num_workers=config.dataloader.num_workers,
    shuffle=False,
)



##############################################################################
# Model, Optimizer, and Scheduler
##############################################################################
model = Sentinel().cuda()

if cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = optim.AdamW(
    model.parameters(),
    lr=config.optimizer.lr,
    weight_decay=config.optimizer.weight_decay,
)

##############################################################################
# Task and Cache
##############################################################################

task = wandb.init(
    name=config.id,
    project="llm-sentinel",
    entity="harshavana",
    id="5gk3khsd",
    resume="allow",
)
# Create a function to copy files to wandb directory
def safe_wandb_save(file_path, wandb_dir):
    dest_path = os.path.join(wandb_dir, os.path.basename(file_path))
    shutil.copy(file_path, dest_path)
    print(f"Copied {file_path} to {dest_path}")

# List of files to save
files_to_save = [
    "detector/t5_sentinel/__init__.py",
    "detector/t5_sentinel/__main__.py",
    "detector/t5_sentinel/dataset.py",
    "detector/t5_sentinel/model.py",
    "detector/t5_sentinel/settings.yaml",
    "detector/t5_sentinel/types.py",
    "detector/t5_sentinel/utilities.py"
]

# Copy each file
for file_path in files_to_save:
    safe_wandb_save(file_path, wandb.run.dir)
# wandb.save("detector/t5_sentinel/__init__.py")
# wandb.save("detector/t5_sentinel/__main__.py")
# wandb.save("detector/t5_sentinel/dataset.py")
# wandb.save("detector/t5_sentinel/model.py")
# wandb.save("detector/t5_sentinel/settings.yaml")
# wandb.save("detector/t5_sentinel/types.py")
# wandb.save("detector/t5_sentinel/utilities.py")

cache = f"storage/{config.id}"
os.path.exists(cache) or os.makedirs(cache)

# if os.path.exists(f"{cache}/state.pt"):
#     state = torch.load(f"{cache}/state.pt")
#     model.load_state_dict(state["model"])
#     optimizer.load_state_dict(state["optimizer"])
#     startEpoch = state["epochIter"] + 1
#     bestValidationAccuracy = state["validAccuracy"]
# else:
#     startEpoch = 0
#     bestValidationAccuracy = float("-inf")
checkpoint_path = "data/checkpoint/T5Sentinel.0613.pt"

if os.path.exists(checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model"])
    print(f"Loaded model checkpoint from {checkpoint_path}")
    # startEpoch = state.get("epochIter", 0) + 1
    startEpoch = 0
    bestValidationAccuracy = state.get("validAccuracy", float("-inf"))
    print(f"Resuming from epoch {startEpoch} with best validation accuracy {bestValidationAccuracy}")
else:
    print("No checkpoint found, starting training from scratch.")
    startEpoch = 0
    bestValidationAccuracy = float("-inf")


##############################################################################
# Training and Validation
##############################################################################
for epoch in range(startEpoch, config.epochs):
    tqdm.write("Epoch {}".format(epoch + 1))
    learnRate = optimizer.param_groups[0]["lr"]
    trainLoss, trainAccuracy = train(model, optimizer, train_loader)
    validAccuracy = validate(model, valid_loader)

    wandb.log(
        {
            "Training Loss": trainLoss,
            "Training Accuracy": trainAccuracy * 100,
            "Validation Accuracy": validAccuracy * 100,
            "Learning Rate": learnRate,
        }
    )

    tqdm.write("Training Accuracy {:.2%}".format(trainAccuracy))
    tqdm.write("Training Loss {:.4f}".format(trainLoss))
    tqdm.write("Validation Accuracy {:.2%}".format(validAccuracy))
    tqdm.write("Learning Rate {:.4f}".format(learnRate))

    checkpoint = {
        "epochIter": epoch,
        "model": model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "validAccuracy": validAccuracy,
    }

    if validAccuracy > bestValidationAccuracy:
        bestValidationAccuracy = validAccuracy
        torch.save(checkpoint, f"{cache}/state.pt")
        tqdm.write("Checkpoint Saved!")

##############################################################################
# Testing
##############################################################################
def plot_confusion_matrix(confusion_matrix, display_labels, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap='crest_r', interpolation='nearest')
    plt.colorbar()
    tick_marks = np.arange(len(display_labels))
    plt.xticks(tick_marks, display_labels, rotation=45)
    plt.yticks(tick_marks, display_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def evaluate_model(model, dataloader):
    model.eval()
    y_pred, y_true = [], []

    with torch.no_grad():
        for i, (corpus_ids, corpus_mask, label_ids) in enumerate(dataloader):
            outputs = model(corpus_ids.cuda(), corpus_mask.cuda())
            predictions = outputs.probabilities.argmax(dim=-1).cpu().numpy()
            labels = label_ids.cpu().numpy()[:, 1]  # Assuming you take care of alignment in the dataset
            
            y_pred.extend(predictions)
            y_true.extend(labels)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=1)
    confusion = confusion_matrix(y_true, y_pred)


    # Optionally print the confusion matrix
    print("Confusion Matrix:\n", confusion[:-1, :-1])
    confusion_mat = confusion[:-1, :-1]

    display_labels = ['Human', 'ChatGPT', 'PaLM', 'LLaMA']
    save_path = "confusion_matrix.png"  # Define the path to save the image
    plot_confusion_matrix(confusion_mat, display_labels, save_path)
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# After completing the training and validation...
test_results = evaluate_model(model, test_loader)
print("Test Metrics:", test_results)



##############################################################################
# Error Analysis
##############################################################################
def error_analysis(model, dataloader):
    model.eval()
    incorrect_predictions = []
    correct_predictions = []

    with torch.no_grad():
        for i, (corpus_ids, corpus_mask, label_ids) in enumerate(dataloader):
            outputs = model(corpus_ids.cuda(), corpus_mask.cuda())
            predictions = outputs.probabilities.argmax(dim=-1).cpu().numpy()
            labels = label_ids.cpu().numpy()[:, 1]  # Assuming you take care of alignment in the dataset
            
            # Iterate over each prediction in the batch
            for idx in range(len(predictions)):
                text = dataloader.dataset.corpus[i * dataloader.batch_size + idx]
                example = {
                    "text": text,
                    "predicted_label": predictions[idx],
                    "true_label": labels[idx]
                }
                
                # Check if the prediction is correct or incorrect
                if predictions[idx] == labels[idx]:
                    correct_predictions.append(example)
                else:
                    incorrect_predictions.append(example)

    return correct_predictions, incorrect_predictions

# Perform error analysis on test data

correct_examples, error_examples = error_analysis(model, test_loader)

# Shuffle the error examples
random.shuffle(error_examples)
# Print a few examples
for example in error_examples[:5]:
    print("Text:", example["text"])
    print("Predicted Label:", example["predicted_label"])
    print("True Label:", example["true_label"])
    print()


# Shuffle the correct prediction examples
random.shuffle(correct_examples)

# Print a few examples of correct predictions
print("Examples of Correct Predictions:")
for example in correct_examples[:5]:
    print("Text:", example["text"])
    print("Predicted Label:", example["predicted_label"])
    print("True Label:", example["true_label"])
    print()
