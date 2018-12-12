"""
This script tests the similarity of the generated annotations against the
initial humanly annotated annotations of Pascal VOC 2012 dataset.

The comparison is made class against class and then averaged over the image.
"""

import glob
import os

import numpy as np
import PIL.Image as Image
import argparse
import collections

from utils.helpers import get_label_info, one_hot_it, reverse_one_hot
from utils.utils import prepare_data, build_images_association_dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', dest='dataset_name', action='store',
                    help='The name of the dataset to analyse.')
parser.add_argument('--ground-truth-masks-directory', dest='ground_truth_masks_directory', action='store',
                    help='The name of the dataset to analyse.')
parser.add_argument('--by-class', dest='by_class', action='store_true',
                    help='Whether or not to compute the average by class.')
args = parser.parse_args()

average_by_class = bool(args.by_class)
dataset_name = str(args.dataset_name)
ground_truth_masks_directory = str(args.ground_truth_masks_directory)

_TRAIN_IMAGES_DIRECTORY = 'train'
_TRAIN_LABELS_DIRECTORY = 'train_labels'

train_images_paths, train_annotations_paths, _, _, _, _ = prepare_data(dataset_dir=dataset_name)
images_association = build_images_association_dictionary(train_images_paths, train_annotations_paths)


def load_image(filename: str):
    """
    Loads an image as numpy array.
    :type filename: str
    """
    img = Image.open(filename)
    data = np.asarray(img, dtype="int32")
    return data


histograms_save_path = os.path.join(dataset_name)

class_names, class_colors = get_label_info(os.path.join(dataset_name, "class_dict.csv"))

def fix_for_marie_cause_dataset_was_broken(class_name):
    if class_name == "tv/monitor":
        return "tv"

    return class_name


def print_table(dictionary):
    ordered_dictionary = collections.OrderedDict(sorted(dictionary.items()))
    print("{:<12} {:<12} {:<3}".format('class_name', 'occurrences', 'similarity'))
    for k in ordered_dictionary.keys():
        print("{:<12} {:<12} {:1.2f}".format(k, ordered_dictionary[k]['occurrences'], ordered_dictionary[k]['score']))


def print_histogram(frequency_array):
    normalized_frequency_array = np.array(frequency_array) / np.sum(frequency_array)
    with open(os.path.join(histograms_save_path, 'similarity.dat'), 'a') as the_file:
        for i in range(len(frequency_array)):
            the_file.write("{} {} {}".format(i, frequency_array[i], normalized_frequency_array[i]))

        print(", ".join(str(x) for x in frequency_array))
        print(", ".join(str(x) for x in normalized_frequency_array))


def class_similarity(class_index, ground_truth, mask):
    mask = reverse_one_hot(one_hot_it(mask, class_colors))
    indices_where_mask_is_class_and_not_important = np.where(mask == class_index)

    b_mask = np.zeros_like(mask)
    b_mask[indices_where_mask_is_class_and_not_important] = 1
    similarity = np.sum(np.logical_and(ground_truth, b_mask)) / np.sum(np.logical_or(ground_truth, b_mask))

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(ground_truth)
    # plt.title("Ground truth for {}".format(class_names[class_index]))
    # plt.subplot(1, 2, 2)
    # plt.imshow(b_mask)
    # plt.title("Sim: {}".format(similarity))
    # plt.show()

    return similarity


if __name__ == '__main__':
    images_count = 0
    method_score = 0

    print('Analysing similarity for dataset at {}'.format(dataset_name))
    print('{} images found.'.format(len(train_images_paths)))

    global_similarity_by_class = {}

    # Holds the entire history of all images' similarity for frequency analysis.
    similarity_history = []

    for image_path, associated_annotations_paths in images_association.items():
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_filename = image_name + ".png"

        try:
            for current_mask_path in associated_annotations_paths:
                images_count += 1

                method_mask = load_image(current_mask_path)

                similarity_for_classes_in_image = {}

                for i in range(len(class_names)):
                    try:
                        ground_truth_mask = load_image(os.path.join(ground_truth_masks_directory,
                                                                    fix_for_marie_cause_dataset_was_broken(
                                                                        class_names[i]),
                                                                    mask_filename))
                        current_class_similarity = class_similarity(i, ground_truth_mask, method_mask)

                        if class_names[i] == "tv/monitor":
                            print('Found monitor')

                        if average_by_class:
                            if class_names[i] in global_similarity_by_class:
                                global_similarity_by_class[class_names[i]]['score'] += current_class_similarity
                                global_similarity_by_class[class_names[i]]['occurrences'] += 1
                            else:
                                global_similarity_by_class[class_names[i]] = {
                                    'score': current_class_similarity,
                                    'occurrences': 1
                                }
                        else:
                            similarity_for_classes_in_image[class_names[i]] = current_class_similarity
                    except FileNotFoundError:
                        pass

                if not average_by_class:
                    mask_average_similarity = 0
                    for class_name in similarity_for_classes_in_image.keys():
                        mask_average_similarity += similarity_for_classes_in_image[class_name]

                    mask_average_similarity /= len(similarity_for_classes_in_image)
                    method_score += mask_average_similarity
                    print('Image #{} ({}) has similarity {}%'.format(images_count,
                                                                     image_name,
                                                                     mask_average_similarity * 100))

                    similarity_history.append(mask_average_similarity * 100)
        except FileNotFoundError:
            print('No mask found for image {}'.format(image_name))

    if average_by_class:
        for class_name in global_similarity_by_class.keys():
            global_similarity_by_class[class_name]['score'] /= global_similarity_by_class[class_name]['occurrences']

        print_table(global_similarity_by_class)
    else:
        method_score /= len(train_annotations_paths)

        print("Dataset {} has average similarity of {}%".format(dataset_name, method_score * 100))

    # Compute histograms.
    histogram, _ = np.histogram(similarity_history)
    print_histogram(histogram)
