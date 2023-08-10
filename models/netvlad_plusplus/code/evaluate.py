import os
import json
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetFeatures, SoccerNetFeaturesTest
from model import Model, NLLLoss
from train import trainer
from inference import model_inference_to_files

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2
from SoccerNet.Evaluation.ActionSpotting import evaluate


def main(args):
    model = Model(input_size=args.feature_dim,
                  num_classes=len(EVENT_DICTIONARY_V2), window_size=args.window_size,
                  vocab_size=args.vocab_size, framerate=args.framerate)

    num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{num_params = }')

    if torch.cuda.is_available():
        model = model.cuda()

    if not args.test_only:
        dataset_train = SoccerNetFeatures(features_path=args.data_path, labels_path=args.labels_path, features=args.features,
                                          split=args.split_train, framerate=args.framerate, window_size=args.window_size)
        dataset_valid = SoccerNetFeatures(features_path=args.data_path, labels_path=args.labels_path, features=args.features,
                                          split=args.split_valid, framerate=args.framerate, window_size=args.window_size)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        loss = NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        trainer(train_loader, val_loader, model, optimizer, scheduler, loss, max_epochs=args.max_epochs)

    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), os.pardir, 'model.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    eval_res_path = os.path.join(os.path.dirname(__file__), os.pardir, 'evaluation_results.json')

    if os.path.isfile(eval_res_path) and False:
        with open(eval_res_path, 'r') as f:
            results = json.load(f)
    else:
        dataset_test = SoccerNetFeaturesTest(features_path=args.data_path, labels_path=args.labels_path, features=args.features,
                                             split=args.split_test, framerate=args.framerate, window_size=args.window_size)

        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)

        games_ouput_path = os.path.join(os.path.dirname(__file__), os.pardir, 'model_outputs', f'split_{args.split_test}')
        model_inference_to_files(test_loader, model, games_ouput_path, nms_window=args.NMS_window, nms_threshold=args.NMS_threshold)

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

    parser.add_argument('--data_path', required=False, type=str, default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data', 'soccernet', 'resnet_features'), help='Path for data')  # '../data/soccernet/resnet_features/'
    parser.add_argument('--labels_path', required=False, type=str, default=os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'data', 'soccernet', 'labels'), help='Path for labels')
    parser.add_argument('--test_only', required=False, type=bool, default=True, help='Perform testing only')
    parser.add_argument('--features',   required=False, type=str, default="ResNET_TF2_PCA512.npy", help='Video features')

    parser.add_argument('--split_train', nargs='+', default=["train", "valid"], help='list of splits for training')
    parser.add_argument('--split_valid', required=False, type=str, default="test", help='split for validation (valid/test)')
    parser.add_argument('--split_test', required=False, type=str, default="test", help='split for testing (test/challenge)')

    parser.add_argument('--feature_dim', required=False, type=int, default=512, help='Number of input features')
    parser.add_argument('--framerate', required=False, type=int, default=2, help='Framerate of the input features')
    parser.add_argument('--window_size', required=False, type=int, default=15, help='Size of the chunk (in seconds)')
    parser.add_argument('--vocab_size', required=False, type=int, default=64, help='Size of the vocabulary for NetVLAD')

    parser.add_argument('--max_epochs', required=False, type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--batch_size', required=False, type=int, default=256, help='Batch size')
    parser.add_argument('--LR', required=False, type=float, default=1e-03, help='Learning Rate')
    parser.add_argument('--patience', required=False, type=int, default=10, help='Patience before reducing LR (ReduceLROnPlateau)')

    parser.add_argument('--NMS_window', required=False, type=int, default=30, help='NMS window in second')
    parser.add_argument('--NMS_threshold', required=False, type=float, default=0.5, help='NMS threshold for positive results')

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    main(args)
