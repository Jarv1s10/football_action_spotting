import os

import torch

from SoccerNet.Evaluation.utils import AverageMeter


def trainer(train_loader, val_loader, model, optimizer, scheduler, loss_func, max_epochs):
    best_loss = 9e99

    for epoch in range(max_epochs):
        print(f'Epoch {epoch} started.')
        best_model_path = os.path.join(os.path.basename(__file__), os.pardir, "model.pth.tar")

        # train for one epoch
        loss_training = train(train_loader, model, loss_func, optimizer, train_mode=True)

        # evaluate on validation set
        loss_validation = train(val_loader, model, loss_func, optimizer, train_mode=False)

        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        state = {
            'epoch': epoch + 1,
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

        print(f'Epoch {epoch} ended. Training loss = {round(loss_training, 4)}, Validation loss = {round(loss_validation, 4)}.')

    return


def train(dataloader, model, loss_func, optimizer, train_mode=False):
    losses = AverageMeter()

    if train_mode:
        model.train()
    else:
        model.eval()

    for feats, labels in dataloader:
        feats = feats.cuda()
        labels = labels.cuda()

        output = model(feats)

        loss = loss_func(labels, output)

        # measure accuracy and record loss
        losses.update(loss.item(), feats.size(0))

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses.avg