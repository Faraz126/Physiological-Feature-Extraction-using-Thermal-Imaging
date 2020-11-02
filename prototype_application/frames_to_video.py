import cv2
import numpy as np
import os
from os.path import isfile, join
sub = 1

pathIn = 'C:/Users/Faraz Ahmed Khan/Desktop/Thermal Repo/Physiological-Feature-Extraction-using-Thermal-Imaging/thermalfaceDB/'
pathOut = 'video.avi'
fps = 4
frame_array = []
def find_sub_frame(file_name):
    sub_ind = file_name.index('sub') + 3
    sub_num = int(file_name[sub_ind: sub_ind + 3])

    seq_ind = file_name.index("seq") + 3
    seq_num = int(file_name[seq_ind: seq_ind + 2])

    frame_ind = file_name.index('frm') + 3
    frame_num = int(file_name[frame_ind: frame_ind + 5])

    return sub_num, seq_num, frame_num


files = [(f, find_sub_frame(f)) for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and f.endswith('.png')]
#files = [i for i in files if i[1][0] == sub]


# for sorting the file names properly
files.sort(key=lambda x: (x[1][1], x[1][2]))
files.sort()
frame_array = []

# for sorting the file names properly

for i in range(500):
    print(i)
    filename = pathIn + files[i][0]
    # reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)

    # inserting the frames into an image array
    frame_array.append(img)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
