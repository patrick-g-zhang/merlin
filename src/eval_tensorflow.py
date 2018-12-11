import os
from run_tensorflow_with_merlin_io import TensorflowClass

if __name__ == '__main__':
    cfg = configuration.cfg
    config_file = "/home/gyzhang/merlin/egs/slt_arctic/s1/conf/acoustic_slt_arctic_full.conf"
    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    tensorflow_instance = TensorflowClass(cfg)
    tensorflow_instance.test_tensorflow_model()