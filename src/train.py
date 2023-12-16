import os
import torch
import pickle
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import seed_everything, result2df, show_result
from dataset import MnistDataset
from optimizer import get_sgd_optimizer, get_adam_optimizer
from models import LinearModel, SimpleConvModel, ResnetModel, ResnetImageNetModel, UNetResnet50

def train(
    train_dataset,
    test_dataset,
    training_epochs = 50,
    image_size = (28, 28),
    batch_size = 10,
    patience = 3,
    n_classes = 10,
    result_dir = "../result",
):  
    # fix seed values
    seed_everything()
    
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset loader
    train_dataloader = DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True)
    # models
    model_dicts = {
        "linear_model":LinearModel(device, image_size, n_classes),
        "simple_conv_model":SimpleConvModel(device, image_size, n_classes),
        "resnet_model":ResnetModel(device, image_size, n_classes),
        "resnet_imagenet_model":ResnetImageNetModel(device, image_size, n_classes),
        "unet_resnet":UNetResnet50(device, image_size, n_classes),
    }

    # optimizers
    optimizer_dicts = {
        "adam":get_adam_optimizer,
        "sgd":get_sgd_optimizer,
    }

    # losses
    loss_dicts = {
        "ce":nn.CrossEntropyLoss(),
    }

    # train
    scheme_results = []
    for model_name, model in model_dicts.items():
        for optimizer_name, optimizer_get_fn in optimizer_dicts.items():
            optimizer = optimizer_get_fn(model)
            for loss_name, loss_fn in loss_dicts.items():
                print(f"model :{model_name}, optimizer:{optimizer_name}, loss_fn:{loss_name}")
                
                # initialize
                epoch_train_accuracies = []
                epoch_train_losses = []
                epoch_valid_accuracies = []
                epoch_valid_losses = []
                patience_stack = 0
                torch.cuda.empty_cache()
                for epoch in range(training_epochs):
                    model.train()
                    preds = []
                    all_labels = []
                    losses = []
                    for i, data in enumerate(train_dataloader):
                        inputs, labels = data
                        inputs = inputs.to(device).to(torch.float)
                        labels = labels.to(device).to(torch.long)
                        # train
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        preds.append(outputs.argmax(axis=1))
                        all_labels.append(labels)
                        losses.append(loss)

                    train_loss = float(torch.stack(losses).mean())
                    train_acc = (torch.cat(preds) == torch.cat(all_labels)).to(torch.float).mean()
                    print(f"train_accuracy:{train_acc:.4f}, train_loss:{train_loss:.4f}", end=" ")
                    epoch_train_accuracies.append(float(train_acc))
                    epoch_train_losses.append(train_loss)

                    # val by test data
                    model.eval()
                    preds = []
                    all_labels = []
                    losses = []
                    torch.cuda.empty_cache()
                    for i, data in enumerate(test_dataloader):
                        
                        inputs, labels = data
                        inputs = inputs.to(device).to(torch.float)
                        labels = labels.to(device).to(torch.long)

                        outputs = model(inputs)
                        loss = loss_fn(outputs, labels)
                        preds.append(outputs.argmax(axis=1))
                        all_labels.append(labels)
                        losses.append(loss)
                        
                    valid_loss = float(torch.stack(losses).mean())
                    valid_acc = (torch.cat(preds) == torch.cat(all_labels)).to(torch.float).mean()
                    print(f"valid_accuracy:{valid_acc:.4f}, valid_loss:{valid_loss:.4f}")
                    epoch_valid_accuracies.append(float(valid_acc))
                    epoch_valid_losses.append(valid_loss)
                    if len(epoch_valid_losses) == 0:
                        continue
                    else:
                        if  min(epoch_valid_losses) < valid_loss:
                            patience_stack += 1
                        else:
                            # save best model
                            
                            
                            weight_path = os.path.join(result_dir, "weights")
                            os.makedirs(weight_path, exist_ok=True)
                            torch.save(model.state_dict(), f"{weight_path}/{model_name}_{optimizer_name}_{loss_name}.pth")
                    if patience_stack >= 3:
                        break
                    
                    

                scheme_results.append(
                    {
                        "model": model_name,
                        "optimizer": optimizer_name,
                        "loss": loss_name,
                        "epoch_train_accuracies": epoch_train_accuracies,
                        "epoch_train_losses": epoch_train_losses,
                        "epoch_valid_accuracies": epoch_valid_accuracies,
                        "epoch_valid_losses": epoch_valid_losses,
                    }
                )
    # save results
    with open(f"{result_dir}/valid_result.pkl", "wb") as f:
        pickle.dump(scheme_results, f)

if __name__ == "__main__":
    data_dir = "../data/MNIST_data/"
    # MNIST dataset
    mnist_train = MnistDataset(root_dir=data_dir,
                            step="train",
                            transform=transforms.ToTensor())

    mnist_test = MnistDataset(root_dir=data_dir,
                            step="test",
                            transform=transforms.ToTensor())
    train(
        mnist_train,
        mnist_test,
        training_epochs = 50,
        image_size = (28, 28),
        batch_size = 10,
        patience = 3,
        n_classes = 10,
        result_dir = "../result",
    )