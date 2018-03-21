import os
import ipdb


lines = open("ensemble.txt", "r").read().splitlines()

image_ids = []
image_sents = []

for idx, line in enumerate(lines):
    if idx % 7 == 0:
        image_ids.append(int(line.split('  ')[1].split('.')[0].split('_')[2]))

    if idx % 7 == 1:
        image_sents.append(line)

f = open("ensemble.json", "w")
f.write('[')
for idx, image_id in enumerate(image_ids):
    if idx != len(image_ids) - 1:
        f.write('{"image_id": ' + str(image_id) + ', "caption": "' + image_sents[idx] + '"}, ')
    else:
        f.write('{"image_id": ' + str(image_id) + ', "caption": "' + image_sents[idx] + '"}]')
