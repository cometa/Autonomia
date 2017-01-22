class DataConfig(object):
#    img_height = 150
    img_height = 90
    img_width = 320
    num_channels = 1
    num_buckets = 15
    
class TrainConfig(DataConfig):
    model_name = "relu"
    batch_size = 64
    num_epoch = 30

class TestConfig(TrainConfig):
    model_path = ""
