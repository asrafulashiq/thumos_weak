from __future__ import print_function
import os
import torch
import torch.optim as optim
import numpy as np

from model import Custom_BMN 
import options

# from model import Model
from test import test_bmn, test
from train import train_bmn
from dataset import Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm


if __name__ == "__main__":
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    print(args)

    dataset = Dataset(args, mode='both')
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter("./logs/" + args.model_name)

    model = Custom_BMN(args)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)

    model_params = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=args.lr)

    init_itr = 1
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.1, verbose=True, min_lr=1e-8
    )

    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        if type(model) == torch.nn.DataParallel:
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if args.test:
        test_bmn(init_itr, dataset, args, model, logger, device)
        raise SystemExit

    best_dmap_itr = (0, init_itr)
    list_loss = []
    for itr in (range(init_itr, args.max_iter)):
        _loss = train_bmn(
            itr, dataset, args, model, optimizer, logger, device
        )
        list_loss.append(_loss)
        if itr % 100 == 0:
            if type(model) == torch.nn.DataParallel:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save(
                {
                    "itr": itr,
                    "model_state_dict": model_state
                },
                "./ckpt/base/" + args.model_name + ".pkl",
            )

            lr_scheduler.step(np.mean(_loss))
            list_loss = []

        if itr % 100 == 0:
            if itr % 500 == 0:
                args.test = True
            print("Iter: {}".format(itr))
            dmap = test(itr, dataset, args, model, logger, device)
            args.test = False

    print("\n\n")
    # print(
    #  f"Best Detection mAP : {best_dmap_itr[0]:.3f} @iter {best_dmap_itr[1]}")
