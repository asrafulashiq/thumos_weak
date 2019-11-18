import argparse

parser = argparse.ArgumentParser(description="WTALC")
parser.add_argument(
    "--lr", type=float, default=0.0002, help="learning rate (default: 0.0001)"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=6,
    help="number of instances in a batch of data (default: 10)",
)
parser.add_argument("--model-name", "-m", default="BMN",
                    help="name to save model")
parser.add_argument(
    "--pretrained-ckpt", "--ckpt", default=None,
    help="ckpt for pretrained model"
)
parser.add_argument(
    "--feature-size", default=2048, help="size of feature (default: 2048)"
)
parser.add_argument("--num-class", default=20,
                    help="number of classes (default: )")
parser.add_argument(
    "--dataset-name", default="Thumos14reduced",
    help="dataset to train on (default: )"
)
parser.add_argument(
    "--max-seqlen",
    default=60,
    type=int,
    help="maximum sequence length during training (default: 750)",
)
parser.add_argument(
    "--Lambda",
    default=0.5,
    type=float,
    help="weight on Co-Activity Loss (default: 0.5)",
)

parser.add_argument(
    "--max-grad-norm",
    type=float,
    default=10,
    help="value loss coefficient (default: 50)",
)
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

parser.add_argument(
    "--max-iter",
    type=int,
    default=10001,
    help="maximum iteration to train (default: 50000)",
)

parser.add_argument("--dis", type=float, default=0.5, help="distance thres")

parser.add_argument("--num-sample", type=float, default=16)

parser.add_argument("--test", action='store_true')

parser.add_argument("--similar-size", type=int, default=2,
                    help="how many instances of similar type will be there")
parser.add_argument(
    "--num-similar",
    default=3,
    type=int,
    help="number of similar pairs in a batch of data  (default: 3)",
)
parser.add_argument("--gamma", type=float, default=1)
parser.add_argument("--gamma2", type=float, default=0.5)

parser.add_argument("--beta1", type=float, default=5)
parser.add_argument("--beta2", type=float, default=1)
parser.add_argument("--thres", type=float, default=0.5)
parser.add_argument("--clip", type=float, default=4)
parser.add_argument("--topk", type=float, default=60)
parser.add_argument("--topk2", type=float, default=10)