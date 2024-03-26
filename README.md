# Environment
In anaconda virtual environment

* Install pytorch appropriate for your environment

```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

* Install other package

```
pip install pandas
pip install matplotlib
pip install opencv-python
pip install tqdm
pip install tensorboard
pip install pyyaml
pip install scipy
```
  
# How to use
## simulation & make dataset
* you can change parameter setting in ```setting.yaml```
* simulation is actuate at ```simulation/force_InvertedPendulum.py```

#### option
```--mode``` : test or train

```--num``` : dataset num. default 1

```--timelength``` : dataset's time length. if 120, dataset's length is 2 minute. default 600

```--Hz``` : default 50

```--animation``` : True or False. visualization. default True

#### example
in simulation folder location,
for make test dataset,

```
python force_InvertedPendulum.py --mode test --num 1 --timelength 100 --Hz 50 --animation False
```

for make train dataset,

```
python force_InvertedPendulum.py --mode train --num 1 --timelength 100 --Hz 50 --animation False
```

datasets are saved in ```train``` folder or ```test``` folder.

## training
* you can change parameter setting in ```setting.yaml```
* ```train.py``` tracing .pt file to use in libtorch (use in c++ environment in robot arm). Extracted traced model is saved in ```extracted_model``` folder.
* Extracted weight is saved in ```weight``` folder.
* run ```train.py``` in terminal

#### option
```--dataset``` : dataset file name

#### example
```
python train.py --dataset train/train_withcontrol3_Hz50.csv
```

## testing
* you can change parameter setting in ```setting.yaml```
  
#### option
```--dataset``` : dataset file name

#### example
```
python test.py --dataset test/test_controlData90%_50Hz_0.csv
```

