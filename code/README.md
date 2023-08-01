## Setup
Clone the repo
```
$ git clone https://github.com/cepdnaclk/e17-4yp-using-machine-learning-in-high-stake-settings.git
```

Our code work is in `code` directory
```
$ cd code
```
### With pip and pipenv
```
$ pip install pipenv
$ pipenv shell
$ pip install -r requirements.txt
```

### With pip and conda
```
$ conda env xai_venv
$ conda activate xai_venv
$ pip install -r requirements.txt
```

### Run pipeline
In the virtual env run,
```
$ python main.py
or
$ python -W ignore main.py
```

## Program arch

config.py : Contains data processing and model training configurattions & constants.

data_processor.py : Contains data processing methods.

feature_engineer.py: Defines the pipeline ans related stuff.

main.py : Executes the data processing, labelling and pipeline

