import PIL.Image as Image

from nsfw import classify

image = Image.open("flower_photos/dandelion/6019234426_d25ea1230a_m.jpg")
sfw, nsfw = classify(image)

print("SFW Probability: {}".format(sfw))
print("NSFW Probability: {}".format(nsfw))