import PIL.Image as Image
import glob
import sys
import argparse
from nsfw import classify



#image = Image.open("flower_photos/dandelion/6019234426_d25ea1230a_m.jpg")
#sfw, nsfw = classify(image)

#print("SFW Probability: {}".format(sfw))
#print("NSFW Probability: {}".format(nsfw))



def main(argv):

    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "--directory",
        help="Path to the root folder for images"
    )

    args = parser.parse_args()

    file_pattern = args.directory + "/*/*.jpg"
    file_list = glob.glob(file_pattern)

    print("Files found: " + str(len(file_list)))

    for image_file in file_list:
        image = Image.open(image_file)
        sfw, nsfw = classify(image)
        print("{},{},{}".format(image_file, sfw, nsfw))

if __name__ == "__main__":
    main(sys.argv)