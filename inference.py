import itertools

import os.path
import torch
from torch.utils.data import Dataset, DataLoader

from .util import AudioUtil


def prepare_single_file_for_inference(fpath, sample_rate, samples_per_chunk, length):
    aud = AudioUtil.open(fpath)
    reaud = AudioUtil.resample(aud, sample_rate)
    rechan = AudioUtil.rechannel(reaud, 2)
    split = torch.split(rechan[0], samples_per_chunk, 1)
    splitSR = list(zip(split, itertools.repeat(sample_rate)))
    splitFull = []

    for s in splitSR:
        dur_aud = AudioUtil.pad_trunc(s, length * 1000)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        splitFull.append(aug_sgram)

    return splitFull


class SingleFileLoader(Dataset):
    def __init__(self, file_path: str, sample_rate: int, samples_per_chunk: int, length: int):
        self.data = prepare_single_file_for_inference(file_path, sample_rate, samples_per_chunk, length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_model(device: torch.device):
    return torch.load(
        os.path.dirname(os.path.realpath(__file__)) + "/songornot_5s.pt",
        map_location=device
    )


def inference(model, file_name: str, device: torch.device, sample_rate: int, samples_per_chunk: int, length: int):
    test_dataset = SingleFileLoader(file_name, sample_rate, samples_per_chunk, length)
    print("Chunks:", len(test_dataset))

    data_loader_test = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    total_prediction = 0
    predictions = []
    # Disable gradient updates
    with torch.no_grad():
        for data in data_loader_test:
            # Get the input features and target labels, and put them on the GPU
            inputs = data.to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            predictions.append(prediction)
            total_prediction += prediction.shape[0]

    print(f'Total items: {total_prediction}')
    preds = torch.cat(predictions)
    x = [list(g) for k, g in itertools.groupby(preds)]
    return [(('song' if y[0].item() == 1 else 'speech'), len(y)) for y in x]
