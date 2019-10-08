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

files_search = os.path.join(args.base_dir, '*.npz')
files = glob.glob(files_search)

files.sort()

print("Found {} files".format(str(len(files))))
print("First file {}".format(files[0]))

for x in range(0,100):
    a = []
    exceptions = 0
    for y in range(0,20):
        index_to_use = (x * 20) + y
        print("Loading {}".format(files[index_to_use]))
        try:
            a.append(np.load(open(files[index_to_use],'rb')))
        except ValueError:
            print('Couldn''t load {} file'.format(files[index_to_use]))
            exceptions += 1
    yahoo = np.concatenate((a[0]['yahoo_nsfw_output'],a[1]['yahoo_nsfw_output']), axis=0)
    mobile_net = np.concatenate((a[0]['MobileNetV2_bottleneck_features'],a[1]['MobileNetV2_bottleneck_features']), axis=0)
    for z in range(2,20-exceptions):
        mobile_net = np.concatenate((mobile_net, a[z]['MobileNetV2_bottleneck_features']), axis=0)
        yahoo = np.concatenate((yahoo, a[z]['yahoo_nsfw_output']), axis=0)
        print('Concatenated {} arrays'.format(str(z)))
    #save b
    print(mobile_net.shape)
    print(yahoo.shape)
    file_name = 'batch-{}.npz'.format((str(x+1)).zfill(2))
    full_path = os.path.join(args.output_dir, file_name)
    np.savez_compressed(full_path, MobileNetV2_bottleneck_features=mobile_net, yahoo_nsfw_output=yahoo)
    print("Saved {}".format(full_path))
