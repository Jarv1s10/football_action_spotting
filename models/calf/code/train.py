import logging
import os
from metrics_visibility_fast import average_mAP, NMS
from SoccerNet.Evaluation.utils import AverageMeter
import time
from tqdm import tqdm
import torch
from preprocessing import batch2long, timestamps2long
from json_io import predictions2json
from SoccerNet.Downloader import getListGames

def trainer(train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            weights,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")

    best_loss = 9e99
    best_metric = -1

    for epoch in range(max_epochs):
        best_model_path = os.path.join(os.path.basename(__file__), os.pardir, "model.pth.tar")

        # train for one epoch
        loss_training = train(
            train_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train = True)

        # evaluate on validation set
        loss_validation = train(
            val_loader,
            model,
            criterion,
            weights,
            optimizer,
            epoch + 1,
            train = False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }

        # Remember best loss and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better and evaluation_frequency > max_epochs:
            torch.save(state, best_model_path)

        # Learning rate scheduler update
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")

        if (prevLR < 2 * scheduler.eps and
                scheduler.num_bad_epochs >= scheduler.patience):
            logging.info(
                "Plateau Reached and no more reduction -> Exiting Loop")
            break
    return

def train(dataloader,
          model,
          criterion,
          weights,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_segmentation = AverageMeter()
    losses_spotting = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    with tqdm(dataloader, total=len(dataloader), ncols=160) as t:
        for feats, labels, targets in t:
            # measure data loading time
            data_time.update(time.time() - end)

            feats = feats.cuda()
            labels = labels.cuda().float()
            targets = targets.cuda().float()

            feats=feats.unsqueeze(1)

            # compute output
            output_segmentation, output_spotting = model(feats)

            loss_segmentation = criterion[0](labels, output_segmentation)
            loss_spotting = criterion[1](targets, output_spotting)

            loss = weights[0]*loss_segmentation + weights[1]*loss_spotting

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))
            losses_segmentation.update(loss_segmentation.item(), feats.size(0))
            losses_spotting.update(loss_spotting.item(), feats.size(0))

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            desc += f'Loss Seg {losses_segmentation.avg:.4e} '
            desc += f'Loss Spot {losses_spotting.avg:.4e} '
            t.set_description(desc)

    return losses.avg
