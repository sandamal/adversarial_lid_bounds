'''
    Author: Sandamal on 22/3/21
    Description: Test FGSM on real datasets
'''
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from scipy.spatial.distance import cdist
from tensorflow import keras
from tensorflow.keras.models import load_model

from utils.lid_utils import get_lids

sys.path.append(os.path.relpath("../utils"))

sns.set()
np.random.seed(2021)

eta_count = 20
eta_min = 0

delta_increment = 0.02
delta_start = 0.02
delta_stop = 2.5

loss_object = tf.keras.losses.CategoricalCrossentropy()

attack = 'fgsm'
dataset = 'mnist'


def create_adversarial_pattern(input_image, input_label, pretrained_model):
    if dataset == 'cifar':
        input_image = input_image.reshape(1, 32, 32, 3)
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = pretrained_model(input_image)
        input_label = input_label.reshape(prediction.shape)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    return gradient


def check_bounds_under_attacks(data, labels, dataset_name, k=1000, nq=50):
    # Dataframes to hold results
    ub_column_names = ['delta', 'perturb_farther', 'lid_b', 'k_a', 'k_c', 'y', 'x', 'cos_psi', 'eta', 'ub']
    ub_df = pd.DataFrame(columns=ub_column_names)

    lb_column_names = ['delta', 'perturb_farther', 'lid_b', 'k_a', 'k_c', 'y', 'x', 'cos_psi', 'eta', 'lb']
    lb_df = pd.DataFrame(columns=lb_column_names)

    original_ub_column_names = ['delta', 'perturb_farther', 'lid_b', 'k_a', 'k_c', 'y', 'x', 'cos_psi', 'eta', 'original_ub']
    original_ub_df = pd.DataFrame(columns=original_ub_column_names)

    original_lb_column_names = ['delta', 'perturb_farther', 'lid_b', 'k_a', 'k_c', 'y', 'x', 'cos_psi', 'eta', 'original_lb']
    original_lb_df = pd.DataFrame(columns=original_lb_column_names)

    closer_count = farther_count = 0
    pretrained_model = load_model(f'models/{dataset_name.upper()}.h5')

    (num_samples, dimensions) = data.shape
    # randomly sample the points to perturb
    idx = np.random.randint(num_samples, size=nq)

    for a_index in idx:
        a = data[a_index, :].reshape(1, dimensions)
        label = labels[a_index]

        # find the distances to its neighbours
        a_distances = cdist(a, data)
        # get the closest k neighbours
        sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=a_distances)[:, 1:k + 1]
        rank_of_c = int(k / 2)
        # rank_of_c = -1
        # select c
        c_index = sort_indices[0, rank_of_c]
        c = data[c_index, :].reshape(1, dimensions)
        # get the vector from a to c
        direction = c - a
        unit_vec_direction = direction / np.linalg.norm(direction)
        # calculate x (distance between a and c)
        x = np.linalg.norm(c - a)

        perturbations = create_adversarial_pattern(a, label, pretrained_model)
        b_direction = perturbations.numpy()
        b_direction = b_direction.reshape(1, -1)
        b_direction = b_direction / np.linalg.norm(b_direction)
        b_direction_angle_cos = np.clip(np.dot(unit_vec_direction, b_direction.T), -1.0, 1.0).squeeze()

        for delta in np.arange(delta_start, delta_stop, delta_increment):
            delta = round(delta, 3)
            perturbation = delta * x
            b = a + perturbation * b_direction
            y = np.linalg.norm(b - c)

            delta_x = delta * x

            # calculate the distance from b to other samples
            distances_from_b = cdist(b, data)
            sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances_from_b)
            k_a = np.where(sort_indices[0] == a_index)[0][0] + 1
            k_c = np.where(sort_indices[0] == c_index)[0][0] + 1
            lid_b = get_lids(b, k=k, batch=data)[0]

            numerator = np.log(k_c / k_a)
            denominator = 1 / delta
            original_denom = y / delta_x

            # set the initial maximum value for eta
            eta_max = (y / delta_x) - 1
            eta_range = np.linspace(eta_min, eta_max, num=eta_count)[1:]

            perturb_closer = perturb_farther = False
            if y < x:
                perturb_closer = True
                closer_count += 1
                if delta >= 2 * b_direction_angle_cos or delta >= y / x:
                    continue
            if y > x:
                perturb_farther = True
                farther_count += 1
                if delta <= 2 * b_direction_angle_cos or delta >= y / x:
                    continue

            ub_dict = []
            lb_dict = []
            original_ub_dict = []
            original_lb_dict = []

            for eta in eta_range:
                try:
                    phi = min(((y + delta * x * eta) / y), (y / (y - delta * x * eta)))
                    eta_ub = lid_b * np.log(phi) / np.absolute(np.log(1 / original_denom))
                    if eta > eta_ub:
                        eta = eta_ub
                    if (1 - (1 / delta)) < eta < ((1 / delta) - 1):
                        ub = numerator / np.log(denominator - eta)
                        ub_dict.append(
                            dict(zip(ub_column_names, [delta, perturb_farther, lid_b, k_a, k_c, y, x, b_direction_angle_cos, eta, ub])))
                except Exception:
                    pass

            for eta in eta_range:
                try:
                    phi = min(((y + delta * x * eta) / y), (y / (y - delta * x * eta)))
                    eta_ub = lid_b * np.log(phi) / np.absolute(np.log(1 / original_denom))
                    if eta > eta_ub:
                        eta = eta_ub
                    if (1 - (1 / delta)) < eta < ((1 / delta) - 1):
                        lb = numerator / np.log(denominator + eta)
                        lb_dict.append(
                            dict(zip(lb_column_names, [delta, perturb_farther, lid_b, k_a, k_c, y, x, b_direction_angle_cos, eta, lb])))
                except Exception:
                    pass

            for eta in eta_range:
                try:
                    phi = min(((y + delta * x * eta) / y), (y / (y - delta * x * eta)))
                    eta_ub = lid_b * np.log(phi) / np.absolute(np.log(1 / original_denom))
                    if eta > eta_ub:
                        eta = eta_ub
                    original_ub = numerator / np.log(original_denom - eta)
                    original_ub_dict.append(
                        dict(zip(original_ub_column_names,
                                 [delta, perturb_farther, lid_b, k_a, k_c, y, x, b_direction_angle_cos, eta, original_ub])))
                except Exception:
                    pass

            for eta in eta_range:
                try:
                    phi = min(((y + delta * x * eta) / y), (y / (y - delta * x * eta)))
                    eta_ub = lid_b * np.log(phi) / np.absolute(np.log(1 / original_denom))
                    if eta > eta_ub:
                        eta = eta_ub
                    original_lb = numerator / np.log(original_denom + eta)
                    original_lb_dict.append(
                        dict(zip(original_lb_column_names,
                                 [delta, perturb_farther, lid_b, k_a, k_c, y, x, b_direction_angle_cos, eta, original_lb])))
                except Exception:
                    pass

            ub_df = ub_df.append(ub_dict, ignore_index=True)
            lb_df = lb_df.append(lb_dict, ignore_index=True)
            original_ub_df = original_ub_df.append(original_ub_dict, ignore_index=True)
            original_lb_df = original_lb_df.append(original_lb_dict, ignore_index=True)

    full_results = f'./results/bound_{dataset_name}_{attack}_k_{k}_nq_{nq}_ub.csv'
    ub_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    ub_df.dropna(subset=["ub"], how="any", inplace=True)
    with open(full_results, 'w+') as f:
        ub_df.to_csv(f, encoding='utf-8', index=False)

    full_results = f'./results/bound_{dataset_name}_{attack}_k_{k}_nq_{nq}_lb.csv'
    lb_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    lb_df.dropna(subset=["lb"], how="any", inplace=True)
    with open(full_results, 'w+') as f:
        lb_df.to_csv(f, encoding='utf-8', index=False)

    full_results = f'./results/bound_{dataset_name}_{attack}_k_{k}_nq_{nq}_original_ub.csv'
    original_ub_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    original_ub_df.dropna(subset=["original_ub"], how="any", inplace=True)
    with open(full_results, 'w+') as f:
        original_ub_df.to_csv(f, encoding='utf-8', index=False)

    full_results = f'./results/bound_{dataset_name}_{attack}_k_{k}_nq_{nq}_original_lb.csv'
    original_lb_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    original_lb_df.dropna(subset=["original_lb"], how="any", inplace=True)
    with open(full_results, 'w+') as f:
        original_lb_df.to_csv(f, encoding='utf-8', index=False)

    print(f'closer - {closer_count}')
    print(f'farther - {farther_count}')
    print('execution complete')


lid_k = 100

if dataset == 'mnist':
    from tensorflow.keras.datasets import mnist

    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = np.vstack((x_train, x_test))

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_train = np.vstack((y_train, y_test))
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_train = x_train.astype('float32') / 255
    check_bounds_under_attacks(x_train, y_train, dataset_name='mnist', k=lid_k)
elif dataset == 'cifar':
    from tensorflow.keras.datasets import cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_train = np.vstack((x_train, x_test))

    y_train = np.vstack((y_train, y_test))
    y_train = keras.utils.to_categorical(y_train)

    x_train = x_train.astype('float32') / 255
    check_bounds_under_attacks(x_train, y_train, dataset_name='cifar', k=lid_k)
