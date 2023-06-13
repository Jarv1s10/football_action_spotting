import os
from contextlib import nullcontext

import torch
from tqdm import tqdm
from SoccerNet.Evaluation.utils import AverageMeter


def epoch(dataloader, model, loss_func, optimizer, train_mode=False, device='cuda'):
    losses = AverageMeter()

    if train_mode:
        model.train()
        context = nullcontext
    else:
        model.eval()
        context = torch.no_grad

    with context():
        for feats_batch, labels_batch in tqdm(dataloader):
            feats_batch = feats_batch.to(device)
            labels_batch = labels_batch.to(device)

            output = model(feats_batch)

            loss = loss_func(labels_batch, output)

            # measure accuracy and record loss
            losses.update(loss.item(), feats_batch.size(0))

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return losses.avg


def trainer(train_loader, val_loader, model, optimizer, scheduler, loss_func, max_epochs):
    best_loss = 9e99

    for ep in range(max_epochs):
        print(f'Epoch {ep} started.')
        best_model_path = os.path.join(os.path.basename(__file__), os.pardir, "model.pth.tar")

        # train for one epoch
        loss_training = epoch(train_loader, model, loss_func, optimizer, train_mode=True)

        # evaluate on validation set
        loss_validation = epoch(val_loader, model, loss_func, optimizer, train_mode=False)

        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        state = {
            'epoch': ep + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        if is_better:
            torch.save(state, best_model_path)

        # Reduce LR on Plateau after patience reached
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)

        # Plateau Reached and no more reduction -> Exiting Loop
        if prev_lr < 2 * scheduler.eps and scheduler.num_bad_epochs >= scheduler.patience:
            break

        print(f'Epoch {ep} ended. Training loss = {round(loss_training, 4)}, Validation loss = {round(loss_validation, 4)}.')
