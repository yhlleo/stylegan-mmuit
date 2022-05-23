import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

mode = "train"  # train or test
attributes_dir = f"./attributes/{mode}"
save_dir  = os.path.join(attributes_dir, "list_attr_ffhq-{}.txt".format(mode))
threshold = 0.6

idx2files = {
    0: "male.txt",
    1: "eyeglasses.txt",
    2: "young.txt",
    3: "smiling.txt"
}

idx2attr ={
    0: "Gender", 
    1: "Glasses", 
    2: "Age", 
    3: "Expression"
}


num_attrs = len(idx2attr)
idx2lists = {i: [] for i in idx2attr}

for i in range(len(idx2attr)):
    with open(os.path.join(attributes_dir, idx2files[i]), 'r') as fin:
        for line in fin:
            splits = line.strip().split("\t")[:2]
            fname  = splits[0]
            label  = 1 - int(float(splits[1]) > threshold) if i!=2 else int(float(splits[1]) > threshold)
            idx2lists[i].append([fname, label])
    print(idx2attr[i], np.sum([v[1] for v in idx2lists[i]]))



# stylegan_ffhq
"""
Male 36235
Smiling 58778
Young 68760
Eyeglasses 13814
"""


new_idx2lists = OrderedDict()
for i in range(len(idx2attr)):
    for it in idx2lists[i]:
        fname, label = it
        if fname not in new_idx2lists:
            new_idx2lists[fname] = []
        new_idx2lists[fname].append(label)

with open(save_dir, "w") as fout:
    fout.write("{}\n".format(len(new_idx2lists)))
    fout.write("\t".join([idx2attr[v] for v in range(len(idx2attr))])+"\n")
    for key in new_idx2lists:
        fout.write("{}\t{}\n".format(key, "\t".join([str(v) for v in new_idx2lists[key]])))

