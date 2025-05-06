import torch
import numpy as np

def train(model, dataloader, loss_fn_original, loss_fn_aug, optimizer, scheduler=None, model_ema=None, device='cuda', grad_accum_steps=1, verbose=False):

    losses = torch.zeros(len(dataloader))

    model.train()

    for batch, (x, y, aug_indices) in enumerate(dataloader):
        x, y, aug_indices = x.to(device), y.to(device), aug_indices.to(device)
        pred = model(x)
        loss_original = loss_fn_original(pred, y)
        loss_aug = loss_fn_aug(pred, y)

        loss = loss_original * ~aug_indices.reshape(-1, 1) + loss_aug * aug_indices.reshape(-1, 1)
        loss = loss.sum()

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch + 1) % grad_accum_steps == 0 or (batch + 1) == len(dataloader):
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                
            model.zero_grad()

            if model_ema is not None:
                model_ema.update(model)

        losses[batch] = loss.detach().cpu()

        if verbose is False:
            print(f'Training... Batch: {batch}/{len(dataloader)}, Train loss: {loss:.4f}', end='\r')
    
    if verbose is False:
        print()

    return losses

def evaluate(model, dataloader, device='cuda', verbose=False):
    num_samples = len(dataloader.dataset)
    num_categories = dataloader.dataset.num_categories
    batch_size = dataloader.batch_size

    preds = torch.zeros((num_samples, num_categories))
    targets = torch.zeros((num_samples, num_categories))

    model.eval()

    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds[batch*batch_size: (batch+1)*batch_size, :] = pred.detach().cpu()
            targets[batch*batch_size: (batch+1)*batch_size, :] = y.detach().cpu()

            print(f'Validating... Batch: {batch}/{len(dataloader)}', end='\r')
        print()

    return preds, targets