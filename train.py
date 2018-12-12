from __future__ import print_function

import argparse
import os
import random
import time
from collections import OrderedDict

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

from utils.naming.naming import FilesFormatterFactory
from utils.utils import build_images_association_dictionary

matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate to use')
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--input_size', type=int, default=512, help='Box six of input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()

input_size = int(args.input_size)
is_dataset_augmented = args.h_flip or args.v_flip or (args.brightness is not None) or (args.rotation is not None)

dataset_name = str(args.dataset)
model_name = str(args.model)
backbone_name = str(args.frontend)

training_parameters = {
    'epochs': int(args.num_epochs),
    'learning_rate': float(args.learning_rate),
    'batch_size': int(args.batch_size),
    'validation_steps': int(args.num_val_images),
    'input_size': input_size,
    'augmented': is_dataset_augmented
}
files_formatter_factory = FilesFormatterFactory(mode='training',
                                                dataset_name=dataset_name,
                                                model_name=model_name,
                                                backbone_name=backbone_name,
                                                training_parameters=training_parameters,
                                                verbose=True,
                                                results_folder='/projets/thesepizenberg/deep-learning/segmentation-suite')


def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.resize_to_size(input_image, output_image, args.input_size)

    if args.h_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0, 1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0 * args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1 * args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.Session(config=config)


# Compute your softmax cross entropy loss
input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3])
output_tensor = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

predictions_tensor, init_fn = model_builder.build_model(model_name=args.model,
                                                        frontend=args.frontend,
                                                        net_input=input_tensor,
                                                        num_classes=num_classes,
                                                        crop_width=args.crop_width,
                                                        crop_height=args.crop_height,
                                                        is_training=True)

weights_shape = (args.batch_size, args.input_size, args.input_size)
unc = tf.where(tf.equal(tf.reduce_sum(output_tensor, axis=-1), 0),
               tf.zeros(shape=weights_shape),
               tf.ones(shape=weights_shape))

loss = tf.reduce_mean(tf.losses.compute_weighted_loss(weights=tf.cast(unc, tf.float32),
                                                      losses=tf.nn.softmax_cross_entropy_with_logits_v2(
                                                          logits=predictions_tensor,
                                                          labels=output_tensor)))

opt = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate,
                                decay=0.995,
                                momentum=0.9).minimize(loss,
                                                       var_list=[var for var in tf.trainable_variables()])

session.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(session)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
checkpoint_formatter = files_formatter_factory.get_checkpoint_formatter(saver=tf.train.Saver(max_to_keep=1000))
summary_formatter = files_formatter_factory.get_summary_formatter()

if args.continue_training:
    print('Loaded latest model checkpoint.')
    checkpoint_formatter.restore(session, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)



print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Front-end -->", args.frontend)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Learning rate -->", args.learning_rate)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices = random.sample(range(0,len(val_input_names)),num_vals)
results_path = "%s/%s/%s" % ("results", args.model, args.frontend)
results_filename = "results-{}-{}-{}.txt".format(args.input_size,
                                                 args.num_val_images,
                                                 'augmented' if is_dataset_augmented else 'non-augmented')

images_association = build_images_association_dictionary(train_input_names, train_output_names)

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):

    current_losses = []

    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))
    number_train_images = len(train_input_names)

    num_iters = len(train_output_names) // args.batch_size
    st = time.time()
    epoch_st = time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = (i * args.batch_size + j) % number_train_images
            id = id_list[index]

            input_image_name = train_input_names[id]
            output_image_name = random.choice(images_association[input_image_name])

            input_image = utils.load_image(input_image_name)
            output_image = utils.load_image(output_image_name)

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)

                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

        # Do the training
        _,current=session.run([opt, loss], feed_dict={input_tensor:input_image_batch, output_tensor:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "[%s - %s] Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (
                args.model, args.frontend, epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)


    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        checkpoint_formatter.save(session=session,
                                  current_epoch=epoch)


    if epoch % args.validation_step == 0:
        print("Performing validation")

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.float32(utils.load_image(val_input_names[ind]))
            gt = utils.load_image(val_output_names[ind])
            input_image, gt = utils.resize_to_size(input_image, gt, desired_size=args.input_size)
            input_image = np.expand_dims(input_image, axis=0) / 255.0
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            valid_indices = np.where(np.sum(gt, axis=-1) != 0)
            gt = gt[valid_indices, :]

            # st = time.time()

            output_image = session.run(predictions_tensor, feed_dict={input_tensor:input_image})


            output_image = np.array(output_image[0,:,:,:])
            output_image = output_image[valid_indices, :]
            output_image = helpers.reverse_one_hot(output_image)

            accuracy, class_accuracies, precision, recall, f1, iou = utils.evaluate_segmentation(pred=output_image,
                                                                                                 label=gt,
                                                                                                 num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            iou_list.append(iou)

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)

        measures = OrderedDict([
            ('epoch', epoch),
            ('accuracy', avg_score),
            ('precision', avg_precision),
            ('recall', avg_recall),
            ('f1', avg_f1),
            ('miou', avg_iou)
        ])

        summary_formatter.update(current_epoch=epoch,
                                 measures_dictionary=measures)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(args.num_epochs-1-epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s!=0:
            train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        utils.LOG(train_time)
        scores_list = []

        # fig1, ax1 = plt.subplots(figsize=(11, 8))
        #
        # ax1.plot(range(epoch+1), avg_scores_per_epoch)
        # ax1.set_title("Average validation accuracy vs epochs")
        # ax1.set_xlabel("Epoch")
        # ax1.set_ylabel("Avg. val. accuracy")
        #
        #
        # plt.savefig(os.path.join(results_path, 'accuracy_vs_epochs.png'))
        #
        # plt.clf()
        #
        # fig2, ax2 = plt.subplots(figsize=(11, 8))
        #
        # ax2.plot(range(epoch+1), avg_loss_per_epoch)
        # ax2.set_title("Average loss vs epochs")
        # ax2.set_xlabel("Epoch")
        # ax2.set_ylabel("Current loss")
        #
        # plt.savefig(os.path.join(results_path, 'loss_vs_epochs_{}.png'.format(args.num_val_images)))
        #
        # plt.clf()
        #
        # fig3, ax3 = plt.subplots(figsize=(11, 8))
        #
        # ax3.plot(range(epoch+1), avg_iou_per_epoch)
        # ax3.set_title("Average IoU vs epochs")
        # ax3.set_xlabel("Epoch")
        # ax3.set_ylabel("Current IoU")
        #
        # plt.savefig(os.path.join(results_path, 'iou_vs_epochs.png'))




