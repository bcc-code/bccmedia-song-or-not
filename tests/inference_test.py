import math, sys
import torch
from inference import inference

SAMPLE_RATE = 44100
LENGTH = 5  # seconds
SAMPLES_PER_CHUNK = SAMPLE_RATE * LENGTH


def main():
    if len(sys.argv) < 3:
        print("Usage: python inference_test.py <path to model> <wav file>")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(sys.argv[1], map_location=torch.device(device))
    model.eval()
    res = inference(model, sys.argv[2], device, SAMPLE_RATE, SAMPLES_PER_CHUNK, LENGTH)

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
