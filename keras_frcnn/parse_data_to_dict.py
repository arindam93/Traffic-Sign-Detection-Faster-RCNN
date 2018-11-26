import cv2
import numpy as np
import os
np.random.seed(2018)

def get_data(input_path):
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}

    with open(input_path, 'r') as f:
        print('Parsing annotation files')

        for line in f:
            line_split = line.strip().split(';')
            # print line_split
            (filename, x1, y1, x2, y2, class_name) = line_split

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg is False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                file_path = 'dataset/png_TrainIJCNN2013/' + filename
                img = cv2.imread(file_path)
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = file_path
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []

                # we use all images as training images
                # tmp = np.random.random_sample()

                all_imgs[filename]['imageset'] = 'train'     # change this to 'test' when using the get_data function for calculating mAP

            all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        # convert the dict all_imgs to a list of dictionaries
        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])


        # make sure the bg class is last in the list
        if found_bg:
            # print True
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping
