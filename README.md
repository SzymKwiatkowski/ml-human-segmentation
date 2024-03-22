# ZPO Project
Clone repo with to include submodules:
```bash
git clone --recurse-submodules https://github.com/SzymKwiatkowski/ml-human-segmentation.git
```

## Usage
To run learning script use it as follows:
```bash
python3 train.py -e 100
```

There is also a way of using neptune-ai by:
- using config file and providing credentials (description below)
- and using two additional script arguments:
  - `-n` or `--use-neptune` - use neptune as logger (by default it is tensorboard)
  - `-c` or `--config` - provide config file path


## Setup
To install requirements run:
```bash
pip install --quiet albumentations lightning monai segmentation-models-pytorch onnx onnxruntime onnxconverter_common torch-pruning
```

## Setup conda
If working with conda use:
```bash
conda create --name ml_segmentation python=3.10.12
```
Then activate conda
```bash
conda activate ml_segmentation
```

```bash
pip install --quiet albumentations lightning monai segmentation-models-pytorch onnx onnxruntime onnxconverter_common torch-pruning
```


## Config
Using template files provided and changing data to corresponding data you can use repository.

`train.py` takes one argument of name config which is yaml file which template is placed in templates directory.

## Using config file
Copy config file to main directory:
```bash
cp templates/config.yaml.template config.yaml
```

Then insert your required credentials