from pathlib import Path
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from hydra.experimental import compose, initialize_config_dir


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path, reference_path, reference_meta_path):
        """Get model method
        Args:
            model_path (str): Path to the trained model directory.
            reference_path (str): Path to the reference data.
            reference_meta_path (str): Path to the meta data.
        Returns:
            bool: The return value. True for success, False otherwise.
        """
        # try:
        root_dir = Path(model_path).parent

        with initialize_config_dir(config_dir=root_dir / 'src' / 'config')):
            cfg = compose(config_name='config.yaml')
            cls.cfg = cfg
        cls.model = load_models(cfg, model_path)[0]
        with open(reference_meta_path) as f:
            reference_meta = json.load(f)
        embeddings, ids = make_reference(
            cfg, reference_path, reference_meta, cls.model)
        cls.embeddings = embeddings
        cls.ids = ids
        return True
        # except:
        #     return False

    @classmethod
    def predict(cls, input):
        """Predict method
        Args:
            input (str): path to the image you want to make inference from
        Returns:
            dict: Inference for the given input.
        """
        # load an image and get the file name
        image = read_image(cls.cfg, input)
        sample_name = os.path.basename(input).split('.')[0]

        # make prediction
        with torch.no_grad():
            prediction = cls.model.extract(image.unsqueeze(0).repeat(32, 1, 1, 1).cuda())[
                0, :].unsqueeze(0).detach().cpu().numpy()

        # make output
        prediction = postprocess(prediction, cls.embeddings, cls.ids)
        output = {sample_name: prediction}

        return output