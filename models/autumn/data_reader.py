import scipy.misc
import random
import csv
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = ''
FILE_EXT = '.jpg'


class DataReader(object):
    def __init__(self, data_dir=DATA_DIR, file_ext=FILE_EXT, sequential=False):
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0
        self.data_dir = data_dir
        self.load()

    # load data as it was loaded from DeepHyperion
    # https://github.com/testingautomated-usi/DeepHyperion/blob/f7f696ba95124125dfe967ea4890d944a9958d77/DeepHyperion-BNG/udacity_integration/train-from-recordings.py#L17
    def load(self):
        """
        Load training data and split it into training and validation set
        """
        tracks = [self.data_dir]

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

        print("Train dataset: " + str(len(X_train)) + " elements")
        print("Test dataset: " + str(len(X_valid)) + " elements")
        self.x_train = X_train
        self.x_valid = X_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.num_train_images = len(self.x_train)
        self.num_val_images = len(self.x_valid)
        self.prev_image = None
        self.last = []

    # total = 0
    # count01 = count005 = count002 = count0 = 0

    # with open('interpolated_center.csv') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         angle = float(row['steering_angle'])
    #         if angle > 0.1 or angle < -0.1 and random.random() > 0.2:
    #             xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
    #             ys.append(row['steering_angle'])
    #             count01 += 1
    #         elif (angle > 0.05 or angle < -0.5) and random.random() > 0.2:
    #             xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
    #             ys.append(row['steering_angle'])
    #             count005 += 1
    #         elif (angle > 0.02 or angle < -0.02) and random.random() > 0.7:
    #             xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
    #             ys.append(row['steering_angle'])
    #             count002 += 1
    #         elif random.random() > 0.8:
    #             xs.append(DATA_DIR + 'training/center/flow_7_cart/' + row['frame_id'] + FILE_EXT)
    #             ys.append(row['steering_angle'])
    #             count0 += 1
    #         total += 1
    #
    # with open('train_center.csv') as f:
    #     reader = csv.DictReader(f)
    #     for row in reader:
    #         angle = float(row['steering_angle'])
    #         xs.append(DATA_DIR + 'Ch2_Train/center/flow_7_local/' + row['frame_id'] + FILE_EXT)
    #         ys.append(row['steering_angle'])
    #         total += 1
    #
    # print('> 0.1 or < -0.1: ' + str(count01))
    # print('> 0.05 or < -0.05: ' + str(count005))
    # print('> 0.02 or < -0.02: ' + str(count002))
    # print('~0: ' + str(count0))
    # print('Total data: ' + str(total))

    # self.num_images = len(xs)
    #
    # c = list(zip(xs, ys))
    # random.shuffle(c)
    # xs, ys = zip(*c)
    #
    # self.train_xs = xs[:int(len(xs) * 0.8)]
    # self.train_ys = ys[:int(len(xs) * 0.8)]
    #
    # self.val_xs = xs[-int(len(xs) * 0.2):]
    # self.val_ys = ys[-int(len(xs) * 0.2):]

    def load_train_batch(self, batch_size):
        x_out = []
        y_out = []
        self.prev_image = None
        self.last = []
        for i in range(0, batch_size):
            # autumn only uses the center images so only use index [0] as image path
            im_path = self.x_train[(self.train_batch_pointer + i) % self.num_train_images][0]
            image = scipy.misc.imread(im_path)
            image = self.process(image)
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.y_train[(self.train_batch_pointer + i) % self.num_train_images]])
        self.train_batch_pointer += batch_size
        return x_out, y_out

    def load_val_batch(self, batch_size):
        x_out = []
        y_out = []
        self.prev_image = None
        self.last = []
        for i in range(0, batch_size):
            # autumn only uses the center images so only use index [0] as image path
            im_path = self.x_valid[(self.val_batch_pointer + i) % self.num_val_images][0]
            image = scipy.misc.imread(im_path)
            image = self.process(image)
            x_out.append(scipy.misc.imresize(image[-400:], [66, 200]) / 255.0)
            y_out.append([self.y_valid[(self.val_batch_pointer + i) % self.num_val_images]])
        self.val_batch_pointer += batch_size
        return x_out, y_out

    def process(self, img):
        img = np.asarray(img)
        prev_image = self.prev_image if self.prev_image is not None else img
        self.prev_image = img
        prev = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
        next = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        self.last.append(flow)

        # weights = [1, 1, 2, 2]
        # last = list(self.last)
        # for x in range(len(last)):
        #     last[x] = last[x] * weights[x]

        avg_flow = sum(self.last) / len(self.last)
        mag, ang = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])

        hsv = np.zeros_like(prev_image)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
