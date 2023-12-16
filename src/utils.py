import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = True  

def result2df(result):
    df = pd.DataFrame(result)

    def best_valid_loss(srs):
        best_model_num = np.array(srs["epoch_valid_losses"]).argmin()
        srs["scheme"] = srs["model"] + "_" + srs["optimizer"] + "_" +srs["loss"]
        srs["train_accuracy"] = srs["epoch_train_accuracies"][best_model_num]
        srs["train_losses"] = srs["epoch_train_losses"][best_model_num]
        srs["valid_accuracies"] = srs["epoch_valid_accuracies"][best_model_num]
        srs["valid_losses"] = srs["epoch_valid_losses"][best_model_num]
        return srs

    df = df.apply(best_valid_loss, axis=1).drop(
        [
            "model",
            "optimizer",
            "loss",
            "epoch_train_accuracies",
            "epoch_train_losses",
            "epoch_valid_accuracies",
            "epoch_valid_losses"
        ]
        , axis=1
    )
    return df

def show_result(df):
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([0,1])
    ax.bar(df["scheme"], df["valid_accuracies"])
    ax.set_xlabel("scheme name")
    ax.set_title("valid accuracy with scheme")
    ax.set_ylabel("accuracy")