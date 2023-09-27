# Gumbel-NeRF: Representing Unseen Objects as Part-Compositional Neural Radiance Fields 

### [Paper]() | [Accompanying video]() | [Checkpoints]()

## Updates

- 2023/09/27: Init repo

## Installation

The main dependencies are in the `requirements.txt`. Please follow instructions in [Switch-NeRF](https://github.com/MiZhenxing/Switch-NeRF/tree/main) to install Tutel if you want to reproduce results of Coded Switch-NeRF.

## Dataset
Please download the ShapeNet-SRN dataset from [this link](https://drive.google.com/drive/folders/1PsT3uKwqHHD2bEEHkIXB99AlIjtmrEiR).

## Reproducing
We used `venv` to build our environment. Please be sure to set `$WORKSPACE` and set the relative paths of dataset and environment correctly before running the following scrips. The hyper-parameters are specified in `switch-nerf/opts.py` and `switch-nerf/models/<method>/configs/<name>.json`
```bash
cd logs

# Generate chunks for training
sh ../job_scripts/generate_chunk_cropped.sh 

# Generate chunks for finetuning
sh ../job_scripts/generate_chunk_nocrop.sh 

# Train
qsub ../job_scripts/<method>/job_train.sh 

# Finetune
qsub ../job_scripts/<method>/job_trainfine.sh 

# Test-time Optimization
qsub ../job_scripts/<method>/job_tto.sh 

# Test
qsub ../job_scripts/<method>/job_test.sh 
```

## Citation

```bibtex

```

## Contact

Any questions, feedback or bug reports are welcomed. Please raise an issue or email to `chingweihsu0809@gmail.com`.

## Liscence


## References

Our code was built upon the following remarkable works.

* [Switch-NeRF](https://github.com/MiZhenxing/Switch-NeRF/tree/main)
* [CodeNeRF](https://github.com/wbjang/code-nerf/tree/main)