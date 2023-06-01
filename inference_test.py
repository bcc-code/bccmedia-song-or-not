import itertools, math, sys
from torch.utils.data import Dataset, DataLoader
import torch
from classifier import AudioClassifier
from util import AudioUtil

SAMPLE_RATE = 44100
LENGTH = 5  # seconds
SAMPLES_PER_CHUNK = SAMPLE_RATE * LENGTH


def prepareSingleFileForInference(fpath, sample_rate, samples_per_chunk):
    aud = AudioUtil.open(fpath)
    reaud = AudioUtil.resample(aud, sample_rate)
    rechan = AudioUtil.rechannel(reaud, 2)
    split = torch.split(rechan[0], samples_per_chunk, 1)
    splitSR = list(zip(split, itertools.repeat(SAMPLE_RATE)))
    splitFull = []

    for s in splitSR:
        dur_aud = AudioUtil.pad_trunc(s, LENGTH * 1000)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        splitFull.append(aug_sgram)

    return splitFull


class SingleFileLoader(Dataset):
    def __init__(self, file_path, sample_rate, samples_per_chunk):
        self.data = prepareSingleFileForInference(file_path, sample_rate, samples_per_chunk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ----------------------------
# Inference
# ----------------------------
def inference2(model, file_name, device, sample_rate, samples_per_chunk):
    test_dataset = SingleFileLoader(file_name, sample_rate, samples_per_chunk)
    print("Chunks:", len(test_dataset))

    data_loader_test = torch.utils.data.DataLoader(
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


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_test.py <path to model> <wav file>")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(sys.argv[1], map_location=torch.device(device))
    model.eval()
    res = inference2(model, sys.argv[2], device, SAMPLE_RATE, SAMPLES_PER_CHUNK)

    start = 0
    end = 0
    current_type = "song"

    for x in res:
        d = x[1]
        # The sensitivity can be adjusted here a bit
        if d <= 2 or current_type == x[0]:
            end += d
        else:
            print("{} ({}): {} - {}".format(current_type, f(end - start), f(start), f(end)))
            start = end
            end = end + x[1]
            current_type = x[0]


def f(sec):
    sec *= LENGTH
    return "{:02.0f}:{:02.0f}".format(math.floor(sec / 60), sec % 60)


if __name__ == "__main__":
    main()
