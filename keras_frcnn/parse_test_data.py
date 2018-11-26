import cv2
import numpy as np
import os
np.random.seed(2018)

def get_data_test(input_path):
    all_imgs = {}


    with open(input_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(';')
            filename = line_split[0]

            if filename not in all_imgs:
                all_imgs[filename] = {}

                file_path = 'dataset/png_TestIJCNN2013/' + filename
                img = cv2.imread(file_path)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = file_path
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows


                all_imgs[filename]['imageset'] = 'test'  


        # convert the dict all_imgs to a list of dictionaries
        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        return all_data, None, None
