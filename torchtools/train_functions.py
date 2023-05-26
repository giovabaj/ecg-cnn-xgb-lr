import numpy as np
import torch
from torch.nn.functional import softmax

from torchtools.early_stopping import EarlyStopping


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Single epoch training"""
    model.train()  # Model in training mode

    train_losses_epoch = []  # list to track the training loss of each batch in the epoch

    for X_train, y_train in train_loader:  # loop over all batches in training
        # Move data and labels to the desired device
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        # reset the gradients
        optimizer.zero_grad()
        # get model predictions
        y_hat = model(X_train)
        # calculate the loss on current batch
        loss = loss_fn(y_hat, y_train)
        # execute the backward pass given the current loss
        loss.backward()
        # update the value of the params
        optimizer.step()
        # append the loss to the losses array
        train_losses_epoch.append(loss.item())

    # Average the loss during epoch
    avg_train_loss = np.average(train_losses_epoch)
    return avg_train_loss


def validation_epoch(model, val_loader, loss_fn, device):
    """Single epoch validation"""
    model.eval()  # Model in validation mode

    val_losses_epoch = []  # to track the training loss as the model trains in validation

    with torch.no_grad():
        for X_val, y_val in val_loader:  # loop over all batches
            # Move data and labels to the desired device
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            # get the predictions from the current state of the model
            y_hat = model(X_val)
            # calculate the loss on the current mini-batch
            loss = loss_fn(y_hat, y_val)
            # append loss
            val_losses_epoch.append(loss.item())

    # Average the loss of the epoch
    avg_val_loss = np.average(val_losses_epoch)
    return avg_val_loss


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, patience, path_res):
    """Train the model"""
    model.to(device)  # Model on device

    # lists to track the average training/validation losses per epoch as the model trains
    avg_train_losses = []
    avg_val_losses = []

    # instantiate the early_stopping class
    path_checkpoint = path_res + 'checkpoint.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_checkpoint, )

    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)

        # Validation
        val_loss = validation_epoch(model, val_loader, loss_fn, device)

        # Adding training and validation losses to the list
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        # Printing metrics for the actual epoch
        print(f"Epoch {epoch + 1} completed. Training loss: {train_loss}, Validation loss: {val_loss}")

        # early stop if validation loss doesn't improve after a given patience
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path_checkpoint))

    # Save best model and training/validation losses
    torch.save(model.state_dict(), path_res + "best_model.pt")
    np.save(path_res + "train_losses", avg_train_losses)
    np.save(path_res + "val_losses", avg_val_losses)

    return model


def model_predictions(model, device, dataloader):
    """Function to make predictions from a trained model.
    Input:
     - model: torch model to make predictions
     - device: device to be used
     - dataloader: torch DataLoader
    Output:
     - y_pred: predictions
    """
    model.eval()  # evaluation mode
    model.to(device)  # putting model on device
    y_pred_list = []  # list with predictions of each batch
    with torch.no_grad():
        for ecg, _ in dataloader:
            ecg = ecg.to(device)
            y_pred_list.append(model(ecg))

    y_pred = torch.cat(y_pred_list)
    y_pred = softmax(y_pred, dim=1)
    return y_pred
