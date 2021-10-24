import os
from PIL import Image
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

directory = r'.\thermalfaceDB'
count = 0
train_images = np.zeros((2935, int(768 / 8), int(1024 / 8)))
keypoints = np.zeros((2935, 68 * 2))
metadeta = {'subj': [], 'frame' : []}
b_box = np.zeros((2935, 4))

for entry in os.scandir(directory):
    if (entry.path.endswith(".png") and entry.is_file()):
        name = entry.path.rstrip('.png')
        image = Image.open(name + '.png')
        resized_image = image.resize((1024//8, 768 // 8))
        #resized_image.show()
        train_images[count] = np.asarray(resized_image)
        with open(name + '.ljson') as json_file:
            data = json.load(json_file)
            x = []
            y = []
            feature_list = []
            for p in data['landmarks']['points']:
                x.append(p[1] / 8 )
                y.append(p[0] / 8)
                feature_list.extend(p)
            keypoints[count] = feature_list
            keypoints[count] /= 8
            b_box[count] = [min(x), max(x), min(y), max(y)]
        name = name.lstrip(r".\thermalfaceDB\ ")
        name = name.split('.')[0]
        name = name.split('_')
        subj = int(name[1].lstrip('sub'))
        frame = int(name[-1].lstrip('frm'))
        metadeta['subj'].append(subj)
        metadeta['frame'].append(frame)


        print(count)
        count += 1


df = pd.DataFrame.from_dict(metadeta)
#df.to_csv('metadeta.csv')
np.save('train_images_small', train_images)
np.save('keypoints_small', keypoints)
np.save('bounding_box_small', b_box)