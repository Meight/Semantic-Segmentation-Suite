from __future__ import print_function, division

import datetime
import glob
import os
import random

import cv2
import numpy as np
import sys
import tensorflow as tf
from PIL import Image
from scipy.misc import imread
from sklearn.metrics import precision_score, \
    recall_score, f1_score

from utils import helpers


def gather_multi_label_data(dataset_directory):
    '''
    Retrieves all images per subset for the training, validation and testing phases.

    For each image, retrieves the associated masks for each class present within the dataset. There may be several
    of such annotations for any given class. The function expects these annotations to have their name beginning
    with the their associated image's name.

    'train':
        'image1.png':
            'background':
                ['mask1', 'mask2']
            'person':
                ['mask1']

    :param dataset_directory: the name of the directory where to look for the subsets.
    :return:
    '''
    paths = {}

    for subset_name in ['train', 'validation', 'test']:
        paths[subset_name] = {}
        subset_annotations_path = subset_name + '_labels'
        cwd = os.getcwd()

        class_directories = os.listdir(os.path.join(cwd, dataset_directory, subset_annotations_path))

        for image_name in os.listdir(os.path.join(cwd, dataset_directory, subset_name)):
            image_path = os.path.join(cwd, dataset_directory, subset_name, image_name)
            image_masks = {}

            for current_class_directory in class_directories:
                current_class_masks = glob.glob(os.path.join(cwd,
                                                             dataset_directory,
                                                             subset_annotations_path,
                                                             current_class_directory,
                                                             os.path.splitext(image_name)[0] + '*'))

                if current_class_masks:
                    image_masks[current_class_directory] = current_class_masks

            paths[subset_name][image_path] = image_masks

    return paths


def get_all_available_annotation_resized_tensors_for_image(input_shape,
                                                           image_masks_dictionary,
                                                           class_colors_dictionary,
                                                           mode='linear'):
    available_modes = ['linear']
    n_hot_encoded_tensors = []

    if not image_masks_dictionary.values():
        return n_hot_encoded_tensors

    if not mode in available_modes:
        raise Exception('Provided mode {} is not supported for tensors generation in multi-label classification.'.
                        format(mode))

    if mode == 'linear':
        # In linear mode, all classes are expected to have the same amount of masks so that one tensor can
        # be created "vertically."
        masks_count = len(next(iter(image_masks_dictionary.values())))
        blank_mask = np.ones(input_shape)

        for k in range(masks_count):
            different_classes_one_hot = []
            for class_name, class_colors in class_colors_dictionary.items():
                class_mask = blank_mask

                if class_name in image_masks_dictionary.keys():
                    mask_image, _ = resize_to_size(load_image(image_masks_dictionary[class_name][k]),  desired_size=input_shape[0])
                    equality = np.equal(mask_image, class_colors)
                    class_mask = np.all(equality, axis=-1)

                print('Mask shape', np.shape(class_mask))
                different_classes_one_hot.append(class_mask)

            if different_classes_one_hot:
                n_hot_encoded_tensors.append(np.stack(different_classes_one_hot, axis=-1))

    return n_hot_encoded_tensors


def to_n_hot_encoded(masks_dictionary, class_names):
    return np.asarray([1 if class_name in masks_dictionary.keys() else 0 for class_name in class_names])


def prepare_data(dataset_dir):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names += glob.glob(cwd + "/" + dataset_dir + "/train/" + file + "*")
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names += glob.glob(cwd + "/" + dataset_dir + "/train_labels/" + file + "*")
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names += glob.glob(cwd + "/" + dataset_dir + "/val/" + file + "*")
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names += glob.glob(cwd + "/" + dataset_dir + "/val_labels/" + file + "*")
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names += glob.glob(cwd + "/" + dataset_dir + "/test/" + file + "*")
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names += glob.glob(cwd + "/" + dataset_dir + "/test_labels/" + file + "*")
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names

def load_image(path):
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
    return image

# Takes an absolute file path and returns the name of the file without th extension
def filepath_to_name(full_name):
    file_name = os.path.basename(full_name)
    file_name = os.path.splitext(file_name)[0]
    return file_name

# Print with time. To console or file
def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Count total number of parameters in the model
def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("This model has %d trainable parameters"% (total_parameters))

# Subtracts the mean images from ImageNet
def mean_image_subtraction(inputs, means=[123.68, 116.78, 103.94]):
    inputs=tf.to_float(inputs)
    num_channels = inputs.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=inputs)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels

def _lovasz_softmax_flat(probas, labels, only_present=True):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.shape[1]
    losses = []
    present = []
    for c in range(C):
        fg = tf.cast(tf.equal(labels, c), probas.dtype) # foreground for class c
        if only_present:
            present.append(tf.reduce_sum(fg) > 0)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
        )
    losses_tensor = tf.stack(losses)
    if only_present:
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    return losses_tensor

def lovasz_softmax(probas, labels, only_present=True, per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    probas = tf.nn.softmax(probas, 3)
    labels = helpers.reverse_one_hot(labels)

    if per_image:
        def treat_image(prob, lab):
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, only_present=only_present)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
    else:
        losses = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), only_present=only_present)
    return losses


def resize_to_size(image, label = None, desired_size = 256):
    if label is not None and ((image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1])):
        raise Exception('Image and label must have the same dimensions! {} vs {}'.format(image.shape, label.shape))

    old_size = image.shape[:2]
    ratio = float(desired_size) / max(old_size)

    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))

    if label is not None:
        label = cv2.resize(label, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [255, 255, 255]
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                               value=color)

    if label is not None:
        label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                   value=color)

    return image, label


def resize_pil_image(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    if image_pil.width / width > image_pil.height / height:
        # It must be fixed by width
        resize_width = width
        resize_height = round(height * (width / image_pil.width))
    else:
        # Fixed by height
        resize_width = round(width * (height / image_pil.height))
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (255, 255, 255, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')


# Randomly crop the image to a specific size. For data augmentation
def random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions! {} vs {}'.format(image.shape, label.shape))

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)

        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape (%d, %d) exceeds image dimensions (%d, %d)!' % (crop_height, crop_width, image.shape[0], image.shape[1]))

# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)

# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT, 
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def filter_valid_entries(prediction, label):
    valid_indices = np.where(label != 255)

    return label[valid_indices], prediction[valid_indices]


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))


    mean_iou = np.mean(I / U)
    return mean_iou


def evaluate_segmentation(pred, label, num_classes, score_averaging="weighted"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    prec = precision_score(flat_pred, flat_label, average=score_averaging)
    rec = recall_score(flat_pred, flat_label, average=score_averaging)
    f1 = f1_score(flat_pred, flat_label, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)

    return global_accuracy, class_accuracies, prec, rec, f1, iou


def compute_class_weights(labels_dir, label_values):
    '''
    Arguments:
        labels_dir(list): Directory where the image segmentation labels are
        num_classes(int): the number of classes of pixels in all images

    Returns:
        class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    image_files = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir) if file.endswith('.png')]

    num_classes = len(label_values)

    class_pixels = np.zeros(num_classes)

    total_pixels = 0.0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        for index, colour in enumerate(label_values):
            class_map = np.all(np.equal(image, colour), axis = -1)
            class_map = class_map.astype(np.float32)
            class_pixels[index] += np.sum(class_map)


        print("\rProcessing image: " + str(n) + " / " + str(len(image_files)), end="")
        sys.stdout.flush()

    total_pixels = float(np.sum(class_pixels))
    index_to_delete = np.argwhere(class_pixels==0.0)
    class_pixels = np.delete(class_pixels, index_to_delete)

    class_weights = total_pixels / class_pixels
    class_weights = class_weights / np.sum(class_weights)

    return class_weights

# Compute the memory usage, for debugging
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # Memory use in GB
    print('Memory usage in GBs:', memoryUse)


def save_image(npdata, out_filename):
    image = Image.fromarray(npdata)
    image.save(out_filename)


def build_images_association_dictionary(input_image_names, output_image_names):
    association_dictionary = {}

    for input_image_name in input_image_names:
        association_dictionary[input_image_name] = [image_name
                                                    for image_name in output_image_names
                                                    if os.path.splitext(os.path.basename(input_image_name))[0]
                                                    in image_name]

    return association_dictionary