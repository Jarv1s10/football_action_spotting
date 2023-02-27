import os
import json
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from dataset import SoccerNetClips, SoccerNetClipsTesting
from model import Model, NLLLoss
from train import trainer
from test import test_spotting

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2


def main(args):
    # create model
    model = Model(input_size=args.feature_dim,
                  num_classes=len(EVENT_DICTIONARY_V2), window_size=args.window_size,
                  vocab_size=args.vocab_size, framerate=args.framerate)

    if torch.cuda.is_available():
        model = model.cuda()

    # create test dataset
    dataset_test = SoccerNetClipsTesting(path=args.data_path, features=args.features, split=args.split_test,
                                         framerate=args.framerate, window_size=args.window_size)

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)

    # run training
    if not args.test_only:
        dataset_train = SoccerNetClips(path=args.data_path, features=args.features, split=args.split_train,
                                       framerate=args.framerate, window_size=args.window_size)
        dataset_valid = SoccerNetClips(path=args.data_path, features=args.features, split=args.split_valid,
                                       framerate=args.framerate, window_size=args.window_size)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        loss = NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience)

        # start training
        trainer(train_loader, val_loader, model, optimizer, scheduler, loss, model_name=args.model_name, max_epochs=args.max_epochs)

    checkpoint = torch.load(os.path.join("../model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])

    eval_res_path = os.path.join('models', args.model_name, 'evaluation_results.json')

    if os.path.isfile(eval_res_path):
        with open(eval_res_path, 'r') as f:
            results = json.load(f)
    else:
        results = test_spotting(test_loader, model=model, model_name=args.model_name, nms_window=args.NMS_window, nms_threshold=args.NMS_threshold)
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

    return


if __name__ == '__main__':

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_path', required=False, type=str, default='../data/dataset_features', help='Path for data')
    parser.add_argument('--features', required=False, type=str, default="ResNET_TF2_PCA512.npy", help='Video features')
    parser.add_argument('--max_epochs', required=False, type=int, default=1000, help='Maximum number of epochs')
    parser.add_argument('--model_name', required=False, type=str, default='NetVLAD++', help='name of the model to save')
    parser.add_argument('--test_only', required=False, type=bool, default=True, help='Perform testing only')

    parser.add_argument('--split_train', nargs='+', default=["train"], help='list of split for training')
    parser.add_argument('--split_valid', nargs='+', default=["valid"], help='list of split for validation')
    parser.add_argument('--split_test', nargs='+', default=["test"], help='list of split for testing')

    parser.add_argument('--feature_dim', required=False, type=int, default=512, help='Number of input features')
    parser.add_argument('--framerate', required=False, type=int, default=2, help='Framerate of the input features')
    parser.add_argument('--window_size', required=False, type=int, default=15, help='Size of the chunk (in seconds)')
    parser.add_argument('--vocab_size', required=False, type=int, default=64, help='Size of the vocabulary for NetVLAD')
    parser.add_argument('--NMS_window', required=False, type=int, default=30, help='NMS window in second')
    parser.add_argument('--NMS_threshold', required=False, type=float, default=0.0, help='NMS threshold for positive results')

    parser.add_argument('--batch_size', required=False, type=int, default=256, help='Batch size')
    parser.add_argument('--LR', required=False, type=float, default=1e-03, help='Learning Rate')
    parser.add_argument('--patience', required=False, type=int, default=10, help='Patience before reducing LR (ReduceLROnPlateau)')

    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    os.makedirs(os.path.join("models", args.model_name), exist_ok=True)

    main(args)
