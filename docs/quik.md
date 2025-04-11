## Install
It is recommended to use anaconda3 to manage and maintain the python library environment.
1. Download the .sh file from the anaconda3 website
2. install anaconda3 with .sh file
```
bash Anaconda3-2023.03-Linux-x86_64.sh
```


### create virtual environment
```
conda create -n ncdia python=3.10 -y
conda activate ncdia
pip install -r requirements.txt
python setup.py install
```

### install package
* pytorch>=1.12.0 torchvision>=0.13.0 (recommand offical torch command)
* numpy>=1.26.4
* scipy>=1.14.0
* scikit-learn>=1.5.1

## Run

### OOD


Example: 
```
bash ./scripts/det_oes_rn18_msp.sh
```



### CIL
Example:
```
bash ./scripts/inc_BM200_lwf.sh
```

