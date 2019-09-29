import numpy as np
import glob
import sys
import argparse
import os

parser = argparse.ArgumentParser('compress data that''s alread been created')

parser.add_argument('--base_dir',
                    help='base directory for bottleneck files')

parser.add_argument('--output_dir', 
                    help='output directory for bottleneck files')

args = parser.parse_args()

output_files_search = os.path.join(args.base_dir, '*.output')
y_true_files_search = os.path.join(args.base_dir, '*.y_true')

output_files = glob.glob(output_files_search)
y_true_files = glob.glob(y_true_files_search)

output_files.sort()
y_true_files.sort()

if len(output_files) != len(y_true_files):
    print("something is wrong")
    sys.exit()

iterator = range(0, len(output_files))
for x in iterator:
    output = np.load(open(output_files[x], 'rb'))
    y_true = np.load(open(y_true_files[x], 'rb'))
    file_name = 'batch-{}.npz'.format((str(x+1)).zfill(4))
    full_path = os.path.join(args.output_dir, file_name)
    np.savez_compressed(full_path, MobileNetV2_bottleneck_features=output, yahoo_nsfw_output=y_true)