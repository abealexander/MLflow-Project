import argparse
import os
import logging
from src.utils.common import read_yaml, create_directories
import tensorflow as tf
import pickle


STAGE = "CALLBACKS" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    params = config["params"]

    logging.info(f"Preparing Callbacks")
    # tensorboard callback
    tensorboard_dir = os.path.join("logs", "tensorboard_logs")
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)

    # early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=params["stopping_patience"], monitor=params["monitor"], restore_best_weights=True)

    # model checkpointing callback
    path_to_ckpt = os.path.join(
        config["data"]["local_dir"],
        config["data"]["model_dir"],
        config["data"]["ckpt_file"])
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(path_to_ckpt, save_best_only=True)

    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, ckpt_cb]
    path_to_callback = os.path.join(
        config["data"]["local_dir"],
        config["data"]["model_dir"],
        config["data"]["callback_file"])
    
    # Pickling
    with open(path_to_callback, "wb") as fp:
        pickle.dump(CALLBACKS_LIST, fp)
    logging.info(f"Callbacks saved to {path_to_callback}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e