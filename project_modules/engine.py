import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Tuple[float, float]:

    """

    Trains the given PyTorch model for an epoch.

    Sets the model to training mode, and then runs it through all the steps. 

    Args:
        model: The PyTorch model to be trained
        dataloader: The DataLoader to be trained on
        loss_fn: The loss function used to minimize
        optimizer: The optimizer used to minimize the loss function
        device: The device to compute on

    Returns:
        A tuple of (train_loss, train_acc)
    
    """

    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1) #  here we go from logits to prediction probabilities to prediction labels 
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # we divide both values because we compute them based on the sum of every batch
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
    ) -> Tuple[float, float]:
    
    """

    Tests the given PyTorch model for an epoch.

    Sets the model to "eval" mode to turn off the random dropout and normalization 
    done in training to tackle overfitting. Then proceeds to run the forward pass

    Args:
        model: The PyTorch model to be tested
        dataloader: The DataLoader to be tested on
        loss_fn: The loss function used to minimize
        device: The device to compute on

    Returns:
        A tuple of (test_loss, test_acc)

    """

    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred_logits = model(X)

            loss = loss_fn(pred_logits, y)
            test_loss = loss.item()

            pred_labels = pred_logits.argmax(dim=1)
            test_acc += ((pred_labels == y).sum().item()/len(pred_labels))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epochs: int,
        device: torch.device,
        writer: torch.utils.tensorboard.writer.SummaryWriter
    ) -> Dict[str, List]:

    """"
    Trains and tests the given PyTorch model.

    Passes the given PyTorch model through the train_step() and test_step()
    functions for a selected number of epochs. Training and testing are done within 
    same epoch loop. Furthermore, it calculates and prints evaluation metrics. 

    Args:
        model: The PyTorch model to be trained and tested
        train_dataloader:The DataLoader to be trained on
        test_dataloader: The DataLoader to be tested on
        optimizer: The optimizer used to minimize the loss function
        loss_fn: The loss function used to minimize
        epochs: How many epochs to train for
        device: The device to compute on

    Returns:
    A dictionary of the metrics per epoch in the form:
            {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    """

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        
        train_loss, train_acc = train_step(
            model=model, 
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.3f} | train_acc: {train_acc:.3f} | test_loss: {test_loss:.3f} | test_acc: {test_acc:.3f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
          writer.add_scalars(main_tag="Loss",
                            tag_scalar_dict={"train_loss": train_loss,
                                              "test_loss": test_loss},
                                              global_step=epoch)
          writer.add_scalars(main_tag="Accuracy",
                            tag_scalar_dict={"train_acc": train_acc,
                                              "test_acc": test_acc},
                                              global_step=epoch)

          writer.close()
        else:
          pass

    return results


