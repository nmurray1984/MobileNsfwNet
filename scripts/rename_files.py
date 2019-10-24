from PIL import Image
import glob
import argparse
import os

IMAGE_SIZE = (224, 224)

parser = argparse.ArgumentParser('Rename images')

parser.add_argument('--base_dir',
                    help='base directory for images')

args = parser.parse_args()

search_path = args.base_dir + "*/*.jpg"
all_files = glob.glob(search_path)
print("Found {} files in directory search path {}".format(len(all_files), search_path))

for file_path in all_files:
    new_file_name = file_path.replace(".jpg", ".original.jpg")
    os.rename(file_path, new_file_name)
    print("Saved file {}".format(new_file_name))



