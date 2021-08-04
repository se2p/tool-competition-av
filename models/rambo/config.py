class DataConfig(object):
    data_path = "./models/rambo/data"
    data_name = "hsv_gray_diff_ch4"
    img_height = 160
    img_width = 320
    num_channels = 2


class TrainConfig(DataConfig):
    model_name = "comma_prelu"
    batch_size = 32
    num_epoch = 10
    val_part = 33
    X_train_mean_path = "./data/X_train_gray_diff2_mean.npy"


class TestConfig(TrainConfig):
    model_path = "./datamodels/weights_hsv_gray_diff_ch4_comma_prelu-03-0.04265.hdf5"
    angle_train_mean = -0.004179079


class VisualizeConfig(object):
    pred_path = "./submissions/final.csv"
    true_path = "./data/CH2_final_evaluation.csv"
    img_path = "./phase2_test/center/*.jpg"