import os
import math

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SoccerNet.Evaluation.utils import AverageMeter
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2


EVENT_DICTIONARY_V2 = dict(EVENT_DICTIONARY_V2)
EVENT_DICTIONARY_V2['Background'] = 17


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def cosine_lr(base_lr, it_max, warmup_iterations):
    def fn(it):
        if it < warmup_iterations:
            return it / warmup_iterations
        return .5 * (np.cos(it / it_max * np.pi) + 1)
    return fn


def train(dataloader, model, optimizer, sheduler, epoch, ts_loss_weight, train_mode=False, device='cuda'):
    losses_meter = AverageMeter()
    class_loss_meter = AverageMeter()
    regr_loss_meter = AverageMeter()

    if train_mode:
        model.train()
        print('Training pass...')
    else:
        print('Validation pass...')
        model.eval()

    it_counter = epoch * len(dataloader)
    for feats, label, rel_offset, *_ in tqdm(dataloader):
        it_counter += 1

        feats = feats.to(device)
        label = label.to(device)
        rel_offset = rel_offset.to(device)

        output, pred_rel_offset = model(feats)
        pred_rel_offset = pred_rel_offset.squeeze(1)

        non_background_indexes_gt = (label != EVENT_DICTIONARY_V2["Background"])

        time_shift_loss = F.mse_loss(pred_rel_offset[non_background_indexes_gt], rel_offset[non_background_indexes_gt].float())
        cl_loss = F.nll_loss(torch.log(output), label)
        loss = cl_loss if math.isnan(time_shift_loss) else cl_loss + ts_loss_weight * time_shift_loss

        losses_meter.update(loss, feats.size(0))
        class_loss_meter.update(cl_loss, feats.size(0))
        regr_loss_meter.update(0 if math.isnan(time_shift_loss) else time_shift_loss, feats.size(0))

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sheduler.step()

    return losses_meter.avg


def trainer(train_loader, val_loader, model, lr, momentum, weight_decay, max_epochs, patience, mse_loss_weight, device='cuda'):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lr(lr, it_max=max_epochs * len(train_loader), warmup_iterations=len(train_loader)))
    early_stopper = EarlyStopper(patience)

    best_loss = 9e99

    for epoch in range(max_epochs):
        if epoch > 0:
            train_loader.dataset.update_background_samples()
        print(f'Epoch {epoch + 1} started.')
        best_model_path = os.path.join(os.path.dirname(__file__), os.pardir, "model.pth.tar")

        # train for one epoch
        loss_training = train(train_loader, model, optimizer, scheduler, epoch, mse_loss_weight, train_mode=True, device=device)

        # evaluate on validation set
        loss_validation = train(val_loader, model, optimizer, scheduler, epoch, mse_loss_weight, train_mode=False, device=device)

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

        print(f'Epoch {epoch} ended. Training loss = {round(loss_training, 4)}, Validation loss = {round(loss_validation, 4)}.')

        if early_stopper.early_stop(loss_validation):
            print(f'Validation loss hasn\'t improved for {patience} epochs, stopping early.')
            break