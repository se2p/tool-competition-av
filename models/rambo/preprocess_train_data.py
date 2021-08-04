from keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import numpy as np
from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv
from sklearn.model_selection import train_test_split
import os

from models.rambo.config import DataConfig


def make_hsv_data(path):
    df = pd.read_csv(path)
    num_rows = df.shape[0]

    X = np.zeros((num_rows, row, col, 3), dtype=np.uint8)
    for i in range(num_rows):
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")

        path = df['fullpath'].iloc[i]
        img = load_img(data_path + path, target_size=(row, col))
        img = img_to_array(img)
        img = rgb_to_hsv(img)
        img = np.array(img, dtype=np.uint8)

        X[i] = img

    return X, np.array(df["angle"])


def make_color_data(path):
    df = pd.read_csv(path)
    num_rows = df.shape[0]

    X = np.zeros((num_rows, row, col, 3), dtype=np.uint8)
    for i in range(num_rows):
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")

        path = df['fullpath'].iloc[i]
        img = load_img(data_path + path, target_size=(row, col))
        img = img_to_array(img)
        img = np.array(img, dtype=np.uint8)

        X[i] = img

    return X, np.array(df["angle"])


def make_grayscale_diff_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]

    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")
        for j in range(num_channels):
            path0 = df['fullpath'].iloc[i - j - 1]
            path1 = df['fullpath'].iloc[i - j]
            img0 = load_img(data_path + path0, grayscale=True, target_size=(row, col))
            img1 = load_img(data_path + path1, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j] = img[:, :, 0]
    return X, np.array(df["angle"].iloc[num_channels:])


def make_grayscale_diff_tx_data(path, num_channels=2):
    df = pd.read_csv(path)
    num_rows = df.shape[0]

    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    for i in range(num_channels, num_rows):
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")
        path1 = df['fullpath'].iloc[i]
        img1 = load_img(data_path + path1, grayscale=True, target_size=(row, col))
        img1 = img_to_array(img1)
        for j in range(1, num_channels + 1):
            path0 = df['fullpath'].iloc[i - j]
            img0 = load_img(data_path + path0, grayscale=True, target_size=(row, col))
            img0 = img_to_array(img0)

            img = img1 - img0
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j - 1] = img[:, :, 0]
    return X, np.array(df["angle"].iloc[num_channels:])


def make_hsv_grayscale_diff_data(path, image_paths, angles, num_channels=2):
    num_rows = min(len(image_paths), len(angles))

    X = np.zeros((num_rows - num_channels, row, col, num_channels), dtype=np.uint8)
    # for i in range(num_channels, num_rows):
    for i in range(num_channels, 500):
        if i % 1000 == 0:
            print("Processed " + str(i) + " images...")
        for j in range(num_channels):
            path0 = image_paths[i - j - 2][0]
            path1 = image_paths[i - j - 1][0]
            img0 = load_img(path0, target_size=(row, col))
            img1 = load_img(path1, target_size=(row, col))
            img0.show()
            img0 = img_to_array(img0)
            img1 = img_to_array(img1)
            img0 = rgb_to_hsv(img0)
            img1 = rgb_to_hsv(img1)
            img = img1[:, :, 2] - img0[:, :, 2]
            img = rescale_intensity(img, in_range=(-255, 255), out_range=(0, 255))
            img = np.array(img, dtype=np.uint8)

            X[i - num_channels, :, :, j] = img
    return X, np.array(angles[num_channels:])


def load_data():
    """
    Load training data and split it into training and validation set
    """
    tracks = [DATA_DIR]

    x = np.empty([0, 3])
    y = np.array([])
    for track in tracks:
        drive = os.listdir(track)
        for drive_style in drive:
            try:
                csv_name = 'driving_log.csv'
                csv_folder = os.path.join(track, drive_style)
                csv_path = os.path.join(csv_folder, csv_name)

                def fix_path(serie):
                    return serie.apply(lambda d: os.path.join(csv_folder, d))

                data_df = pd.read_csv(csv_path)
                pictures = data_df[['center', 'left', 'right']]
                pictures_fixpath = pictures.apply(fix_path)
                csv_x = pictures_fixpath.values

                csv_y = data_df['steering'].values
                x = np.concatenate((x, csv_x), axis=0)
                y = np.concatenate((y, csv_y), axis=0)
            except FileNotFoundError:
                print("Unable to read file %s" % csv_path)
                exit()

    try:
        X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)
    except TypeError:
        print("Missing header to csv files")
        exit()

    # print("Train dataset: " + str(len(X_train)) + " elements")
    # print("Test dataset: " + str(len(X_valid)) + " elements")
    return X_train, X_valid, y_train, y_valid


DATA_DIR = './training_recordings'

if __name__ == "__main__":
    config = DataConfig()
    data_path = config.data_path
    row, col = config.img_height, config.img_width

    X_train, X_valid, y_train, y_valid = load_data()

    print("Pre-processing phase 1 data...")
    X_train_gray_diff, y_train_gray_diff = make_hsv_grayscale_diff_data("data/train_round1.txt", X_train, y_train, 2)
    np.save(data_path + "/X_train_round1_hsv_gray_diff_ch4", X_train_gray_diff)
    np.save(data_path + "/y_train_round1_hsv_gray_diff_ch4", y_train_gray_diff)

    X_val_gray_diff, y_val_gray_diff = make_hsv_grayscale_diff_data("data/val_round1.txt", X_valid, y_valid, 2)
    np.save("{}/X_train_round1_hsv_gray_diff_ch4".format(data_path), X_val_gray_diff)
    np.save("{}/y_train_round1_hsv_gray_diff_ch4".format(data_path), y_val_gray_diff)

    print("Pre-processing phase 2 data...")
    for i in range(1, 6):
        # X_train, y_train = make_hsv_grayscale_diff_data("data/train_round2_part" + str(i) + ".txt", X_train, y_train,
        #                                                 2)
        np.save("{}/X_train_round2_hsv_gray_diff_ch4_part{}".format(data_path, i), X_train_gray_diff)
        np.save("{}/y_train_round2_hsv_gray_diff_ch4_part{}".format(data_path, i), y_train_gray_diff)
