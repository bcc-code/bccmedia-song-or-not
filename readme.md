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


