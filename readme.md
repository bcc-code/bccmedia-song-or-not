# Song-Or-Not
## Summary

This is the code used to train a neural network with the goal of detecting of parts of a recording
are speech or song.

## Training data

The training data used is in forms of MP3 files (not included) and can be placed into the `songs`
or `speech` folders. Then you can run the `./split.sh` scripts which splits the file into chunks
with the specified length. Currently only files with 44100 Hz sample rate are supported

## Dependencies

Incomplete and untested list:

```
conda install -c apple tensorflow-deps 
conda install tensorflow_io
conda install torchaudio
```

## Training

Run `python3 train.py`

This will produce a `./songornot_trained.pt` file. If you run the script again the file will be replaced

## Inference

As a sample you can run `./inference_test.py <model> <audio file>`.

Results will be similar to this:

```
/Users/matjaz/meeting.wav
Chunks: 1317
Total items: 1317
song (15:10): 00:00 - 15:10
speech (04:00): 15:10 - 19:10
song (03:05): 19:10 - 22:15
speech (18:30): 22:15 - 40:45
song (04:00): 40:45 - 44:45
speech (00:20): 44:45 - 45:05
song (00:15): 45:05 - 45:20
speech (08:05): 45:20 - 53:25
song (03:30): 53:25 - 56:55
speech (25:05): 56:55 - 82:00
song (01:55): 82:00 - 83:55
speech (20:40): 83:55 - 104:35
song (03:30): 104:35 - 108:05

```

