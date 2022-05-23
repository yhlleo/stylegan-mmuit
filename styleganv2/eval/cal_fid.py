
import os
import sys
import glob
import argparse 

sys.path.append(".")
sys.path.append("..")

from metrics.fid import calculate_fid_given_lists

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str)
parser.add_argument('--trg_dir', type=str)
parser.add_argument('--index', type=int, default=0, help='0,1,2,3,4')
args = parser.parse_args()


source_list = glob.glob(os.path.join(args.src_dir, '*.png'))
target_list = glob.glob(os.path.join(args.trg_dir, '*-{}-*.jpg'.format(args.index) if args.index != -1 else "*.jpg"))

print(calculate_fid_given_lists([source_list, target_list]))