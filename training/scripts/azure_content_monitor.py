import glob
import argparse
import os
import urllib
import requests
import json

parser = argparse.ArgumentParser('run images through azure and get labels to stdout')

parser.add_argument('--base_dir',
                    help='base directory for images')

parser.add_argument('--subscription_key', 
                    help='subscription key for api')

args = parser.parse_args()

files_search = os.path.join(args.base_dir, '*.jpg')
files = glob.glob(files_search)

headers = {
    # Request headers
    'Content-Type': 'image/jpeg',
    'Ocp-Apim-Subscription-Key': args.subscription_key,
}

url = 'https://southcentralus.api.cognitive.microsoft.com/contentmoderator/moderate/v1.0/ProcessImage/Evaluate?CacheImage=false'

for image in files:
    with open(image, 'rb') as f:
        response = requests.post(url, headers=headers, data=f)
        text = json.loads(response.text)
        try:
            print('{},{},{},{},{},{}'.format(
                image,
                text['Status']['Description'],
                text['IsImageRacyClassified'],
                text['RacyClassificationScore'],
                text['IsImageAdultClassified'],
                text['AdultClassificationScore'],
            ))
        except KeyError:
            pass
