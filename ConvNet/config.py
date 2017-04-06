class DataConfig(object):
#    img_height = 150
#    img_height = 90
#    img_width = 320

    # 1 channel Y or 3 channels YCrCb
    num_channels = 1
    num_buckets = 1
    # number of frames to skip ahead in matching telemetry
    skip_ahead = 1

    # image y-axis cropping 
    img_yaxis_start = 140
    img_yaxis_end = 227
    img_height = img_yaxis_end - img_yaxis_start + 1
    ycrop_range = [140, -20]
    # image x-axis cropping
    img_xaxis_start = 0
    img_xaxis_end = 319
    img_width = img_xaxis_end - img_yaxis_start + 1
    # image resampling dimensions
    img_resample_dim = (128,128)
    cspace = 'YCR_CB' #image color space to be fed to model
    keep_rate = 0.5 #Dropout 
    reg_fc = 0.05 #regularizer FC layers
    reg_conv = 0.00001 #regularizer Conv layers 


class TrainConfig(DataConfig):
    model_name = "relu"
    batch_size = 128
    num_epoch = 50
    validation_split = 0.2
    model = 'model_wroscoe_mod'
    model_type = 'regression'
    data_augmentation = 5
    seed = 42

class TestConfig(TrainConfig):
    model_path = ""
  
