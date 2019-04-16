from __future__ import print_function
import os
import torch
import torch.optim as optim

from model import Model as Model
import options_expand as options

# from model import Model
from test2 import test

from train_expand import train
from dataset import Dataset
from tensorboard_logger import Logger


torch.set_default_tensor_type("torch.cuda.FloatTensor")

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
    logger = Logger("./logs/" + args.model_name)

    model = Model(dataset.feature_size, dataset.num_class)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr,
    #                       weight_decay=0.0005)
    init_itr = 0
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [3000, 5000, 10000], 0.5
    )

    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        if type(model) == torch.nn.DataParallel:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        init_itr = checkpoint["itr"]

    if args.test:
        test(init_itr, dataset, args, model, logger, device)
        raise SystemExit

    best_dmap_itr = (0, init_itr)
    for itr in range(init_itr, args.max_iter):
        train(
            itr, dataset, args, model, optimizer, logger, device,
            scheduler=None
        )
        if itr % 100 == 0 and not itr == 0:
            if type(model) == torch.nn.DataParallel:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()
            torch.save(
                {
                    "itr": itr,
                    "model_state_dict": model_state
                    # 'optimizer_state_dict': optimizer.state_dict()
                },
                "./ckpt/thumos/" + args.model_name + ".pkl",
            )
        if itr % 100 == 0 and not itr == 0:
            if itr % 500 == 0 and not itr == 0:
                args.test = True
            dmap = test(itr, dataset, args, model, logger, device)

            args.test = False

    # print()
    # print(
    #  f"Best Detection mAP : {best_dmap_itr[0]:.3f} @iter {best_dmap_itr[1]}")
