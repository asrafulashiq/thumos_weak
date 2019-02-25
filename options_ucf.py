import argparse

parser = argparse.ArgumentParser(description='WTALC')
parser.add_argument('--lr', type=float, default=0.005,help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--model-name', default='ucf', help='name to save model')
parser.add_argument('--pretrained-ckpt', default='./ckpt/ucf.pkl', help='ckpt for pretrained model')
parser.add_argument('--feature-size', default=4096, help='size of feature (default: 2048)')
parser.add_argument('--num-class', default=13, help='number of classes (default: )')
parser.add_argument('--dataset-name', default='UCF_Crime', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', default=100, help='maximum sequence length during training (default: 750)')
parser.add_argument('--Lambda', default=0.5, help='weight on Co-Activity Loss (default: 0.5)')
parser.add_argument('--num-similar', default=3, help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--max-grad-norm', type=float, default=10, help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-iter', type=int, default=30000, help='maximum iteration to train (default: 50000)')
parser.add_argument('--valid', action='store_true', help='validation')


