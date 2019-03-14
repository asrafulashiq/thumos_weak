from __future__ import print_function
import argparse
import os
import torch
import torch.optim as optim
from model import Model_tcn as Model
from video_dataset import Dataset
from test_detect import test
from train_detect import train
from tensorboard_logger import Logger
import options_attn as options
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if __name__ == '__main__':
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    dataset = Dataset(args)
    if not os.path.exists('./ckpt/'):
        os.makedirs('./ckpt/')
    if not os.path.exists('./logs/' + args.model_name):
        os.makedirs('./logs/' + args.model_name)
    logger = Logger('./logs/' + args.model_name)

    model = Model(dataset.feature_size, dataset.num_class).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.005)
    init_itr = 0

    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_itr = checkpoint['itr']

    for itr in range(init_itr, args.max_iter):
        train(itr, dataset, args, model, optimizer, logger, device,
              valid=args.valid)
        if itr % 100 == 0 and not itr == 0:
            torch.save({
                'itr': itr,
                'model_state_dict': model.state_dict()
                # 'optimizer_state_dict': optimizer.state_dict()
            }, './ckpt/' + args.model_name + '.pkl')
        if itr % 50 == 0 and not itr == 0:
            test(itr, dataset, args, model, logger,
                 device, is_detect=True, is_score=False)
