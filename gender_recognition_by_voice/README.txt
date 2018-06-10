本次实验在Anaconda3的虚拟环境内，除了XGBoost之外无需额外安装


- 安装XGBoost Python pakage的方法（基于Ubuntu系统）

git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
sudo apt-get install python-setuptools
cd python-package; sudo python setup.py install
export PYTHONPATH=~/xgboost/python-package

大致运行时间：5分钟以内（单核CPU）
