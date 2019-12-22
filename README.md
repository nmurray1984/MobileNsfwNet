# Try it out

If you'd like to to try it out, install the [Chrome Extension](https://chrome.google.com/webstore/detail/porn-blocker/jdjldnifikfcmahncbamfcglceiegbje?hl=en)

# MobileNsfwNet

This project is meant to further the availability of NSFW detection on all devices. Porn is available instantly just about anywhere - phones, tablets, TVs, gaming consoles, etc. Unfortunately, for those don't want to see it or keep others from seeing it, the tools availible are unreliable or easy to get around. We are striving to solve this problem by building the following:

1) A neural network that accurately detects pornographic images.

2) Continually experiment to reduce the processing power required for inference

3) Release it for free

# How it's built

The original training dataset was built using tools provided by [####](https://github.com/alex000kim/nsfw_data_scraper). Each image was then verified as NSFW by running it through Azure's Content Moderator service. Those classifications where then used to train a MobileNetv2 classifier (chosen due to its small size).

# Where is the dataset?

Due to the sensitivity of the dataset, we cannot release it. Given there is significant public utility in providing access so that others can build their own classifiers, we have released the MobileNetv2 bottlenecks on [Kaggle](https://www.kaggle.com/nmurray1234/yahoo-nsfw-as-mobilenetv2-bottlenecks). These are the result of inference through all but the top layers of MobileNetv2, so that those that are interested in applying transfer learning to the problem have a dataset to work on, without releasing the images themselves.

# Results

In general, the network works well on pornographic images, reaching 95% accuracy against how Azure's Content Moderation service classified a set of 40K images. When using the Chrome Extension, it is apparent that there are still many false positives identified. We've added a reporting option so that those using the Chrome Extension can tell us if an image should not be flagged. Our objective is then to train with the control set so that false positives are less likely.
