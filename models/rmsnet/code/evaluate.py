import os
import json
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetFeatures
from model import RMSNet
from train import cosine_lr, trainer
from inference import model_inference_to_files

from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2


def main(args):
    model = RMSNet(args.feature_dim, len(EVENT_DICTIONARY_V2) + 1)

    # num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for n, p in model.named_parameters():
        print(n, p.numel())
    # print(f'{num_params = }')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    checkpoint_path = os.path.join(os.path.dirname(__file__), os.pardir, 'model.pth.tar')
    if args.test_only or args.start_from_checkpoint and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

    if not args.test_only:
        dataset_train = SoccerNetFeatures(features_path=args.features_path, labels_path=args.labels_path, features=args.features,
                                          split=args.split_train, frames_per_clip=args.frames_per_clip, framerate=args.framerate,
                                          class_samples_per_epoch=args.class_samples_per_epoch, test_overlap=args.test_overlap)
        dataset_valid = SoccerNetFeatures(features_path=args.features_path, labels_path=args.labels_path, features=args.features,
                                          split=args.split_valid, frames_per_clip=args.frames_per_clip, framerate=args.framerate,
                                          class_samples_per_epoch=args.class_samples_per_epoch, test_overlap=args.test_overlap)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        trainer(train_loader, val_loader, model, args.lr, args.momentum, args.weight_decay, args.max_epochs, args.patience, args.mse_loss_weight, device=device)

    eval_res_path = os.path.join(os.path.dirname(__file__), os.pardir, 'evaluation_results.json')

    if os.path.isfile(eval_res_path):
        with open(eval_res_path, 'r') as f:
            results = json.load(f)

    else:
        dataset_test = SoccerNetFeatures(features_path=args.features_path, labels_path=args.labels_path, features=args.features,
                                          split=args.split_test, frames_per_clip=args.frames_per_clip, framerate=args.framerate,
                                          class_samples_per_epoch=args.class_samples_per_epoch, test_overlap=args.test_overlap)

        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)

        games_ouput_path = os.path.join(os.path.dirname(__file__), os.pardir, 'model_outputs', f'split_{args.split_test}')
        model_inference_to_files(test_loader, model, games_ouput_path, args.nms_window_ms, device=device)

        results = evaluate(args.labels_path, games_ouput_path, split=args.split_test, framerate=args.framerate)

        results['a_mAP_per_class'] = {INVERSE_EVENT_DICTIONARY_V2[i]: val for i, val in enumerate(results['a_mAP_per_class'])}
        results['a_mAP_per_class_visible'] = {INVERSE_EVENT_DICTIONARY_V2[i]: val for i, val in enumerate(results['a_mAP_per_class_visible'])}
        results['a_mAP_per_class_unshown'] = {INVERSE_EVENT_DICTIONARY_V2[i]: val for i, val in enumerate(results['a_mAP_per_class_unshown'])}

        with open(eval_res_path, 'w') as f:
            json.dump(results, f, indent=4)

    a_mAP = results["a_mAP"]
    a_mAP_per_class = results["a_mAP_per_class"]

    print('-'*100)
    print('TEST EVALUATION METRICS:\n')
    print(f"a_mAP all: {a_mAP}")
    print(f"a_mAP all per class: {a_mAP_per_class}")

if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--features_path', required=False, type=str, default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data', 'soccernet', 'resnet_features'), help='Path for resnet features')
    parser.add_argument('--labels_path', required=False, type=str, default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data', 'soccernet', 'labels'), help='Path for labels')
    parser.add_argument('--test_only', required=False, type=bool, default=False, help='Perform testing only')
    parser.add_argument('--start_from_checkpoint', required=False, type=bool, default=True, help='Start training from saved checkpoint or freshly initialize model')
    parser.add_argument('--features',   required=False, type=str, default="ResNET_TF2_PCA512.npy", help='Video features')

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of splits for training')
    parser.add_argument('--split_valid', required=False, type=str, default= "valid", help='split for validation (valid/test)')
    parser.add_argument('--split_test', required=False, type=str, default="test", help='split for testing (test/challenge)')

    parser.add_argument('--feature_dim', required=False, type=int, default=512, help='Number of input features')
    parser.add_argument('--framerate', required=False, type=int, default=2, help='Framerate of the input features')

    parser.add_argument('--max_epochs', required=False, type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--batch_size', required=False, type=int, default=64, help='Batch size')
    parser.add_argument('--lr', required=False, type=float, default=1e-03, help='Learning rate')
    parser.add_argument('--momentum', required=False, type=float,  default=0.9, help='momentum')
    parser.add_argument('--weight-decay', required=False, default=1e-4, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument('--frames_per_clip', required=False, default=41, type=int, help='Duration (in frames) of each clip')
    parser.add_argument('--mse_loss_weight', required=False, default=10, type=int, help='Time shift MSE loss weight')
    parser.add_argument('--nms_window_ms', required=False, default=2000, type=int, help='ms for NMS (remove duplicate predictions with distance < nms ms)')
    parser.add_argument('--test_overlap', required=False, default=0, type=float, help='percentage in [0,1] of overlap between consecutive clips during inference')
    parser.add_argument('--class_samples_per_epoch', required=False, default=1000, type=int, help='number of random samples for each class in a training epoch')
    parser.add_argument('--patience', required=False, type=int, default=5, help='Patience before reducing LR (ReduceLROnPlateau)')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    main(args)
