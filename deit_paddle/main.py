from pathlib import Path
import argparse
from deit_paddle.get_argument import get_args_parser

from deit_paddle.datasets import build_dataset, DataLoader
from paddle.io import RandomSampler


parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
if args.output_dir:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

dataset_test, args.nb_classes = build_dataset(is_train=False, args=args)
sampler_test = RandomSampler(dataset_test)
data_loader_test = DataLoader(
    dataset_test, sampler=sampler_test,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True
)

for img in data_loader_test():
    print(img[0].shape)
    break