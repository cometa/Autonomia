class DataConfig(object):
#    img_height = 150
#    img_height = 90
#    img_width = 320

    # 1 channel Y or 3 channels YCrCb
    num_channels = 1
    num_buckets = 15
    # number of frames to skip ahead in matching telemetry
    skip_ahead = 1

    # image y-axis cropping 
    img_yaxis_start = 140
    img_yaxis_end = 227
    img_height = img_yaxis_end - img_yaxis_start + 1
    # image x-axis cropping
    img_xaxis_start = 0
    img_xaxis_end = 319
    img_width = img_xaxis_end - img_yaxis_start + 1
    # image resampling dimensions
    img_resample_dim = (180,180)
    
class TrainConfig(DataConfig):
    model_name = "relu"
    batch_size = 64
    num_epoch = 10
    validation_split = 0.3
    model = 'model_2softmax'

class TestConfig(TrainConfig):
    model_path = ""
