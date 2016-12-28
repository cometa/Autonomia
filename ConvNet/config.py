class DataConfig(object):
    data_path = "/home/oem/Autonomia-Video/1482616994"
    data_name = "hsv_gray"
    img_height = 240
    img_width = 320
    num_channels = 1
    num_buckets = 15
    
class TrainConfig(DataConfig):
    model_name = "prelu"
    batch_size = 32
    num_epoch = 10

class TestConfig(TrainConfig):
    model_path = ""
