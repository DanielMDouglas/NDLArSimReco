# NDLArSimReco

Making a CNN for mapping `larnd-sim` outputs back to their `edep-sim` counterparts.  Work in progress!

## Manifests

The network and its parameters are described in a yaml file in `NDLArSimReco/manifests`.  You can see an example manifest which I use on my local machine in `NDLArSimReco/manifests/localUResNetManifest.yaml`.

## Training

Training the network can be done using the `train.py` script, which will initialize the network based on a supplied manifest, and train it against the training data (also described in the manifest).

```
usage: train.py [-h] [-m MANIFEST] [-f] [-v] [-c CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -m MANIFEST, --manifest MANIFEST
                        network manifest yaml file
  -f, --force           forcibly train the network from scratch
  -v, --verbose         print extra debug messages
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        checkpoint file to start from
```

During the training process, a checkpoint is saved every 1/10th of an epoch.  This checkpoint system is in the middle of an overhaul, so things are changing quickly here, but in general, you can use these checkpoints to resume training from a certain point.  It's generally best to resume from the latest epochal checkpoint, since the data loading is not 100% repeatable yet.

## Testing

After training, the script at `NDLArSimReco/eval.py` will use each of the epochal checkpoints and assess the loss against 50 batches of the test data.  This loss data is saved in a plaintext file, `testEval.dat` in the output directory.

```
usage: eval.py [-h] [-m MANIFEST] [-v] [-t] [-n NBATCHES] [-l]

optional arguments:
  -h, --help            show this help message and exit
  -m MANIFEST, --manifest MANIFEST
                        network manifest yaml file
  -v, --verbose         print extra debug messages
  -t, --trainMode       run the evaluation loop in train mode instead of eval mode (useful for networks
                        with dropout)
  -n NBATCHES, --nBatches NBATCHES
                        Number of batches from the test dataset to evaluate on each checkpoint
  -l, --useLast         optionally, use the last checkpoint in the epoch as a proxy
```

## Analysis

Some simple analysis can be found in `NDLArSimReco/analysis/errorAnalysis.py`:

```
usage: errorAnalysis.py [-h] [-m MANIFEST] [-c CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -m MANIFEST, --manifest MANIFEST
                        network manifest yaml file
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        checkpoint file to use
```