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
```
  
# How to use
## simulating & make dataset
* simulation is actuate at ```simulation/force_InvertedPendulum.py```

## training
* you can change parameter setting in ```setting.yaml```
* ```train.py``` also tracing .pt file to use in libtorch (use in c++ environment in robot arm). Extracted traced model is saved in ```extracted_model``` folder.
* run ```train.py``` in terminal
  
```
python train.py
```

## testing
```
python test.py
```

