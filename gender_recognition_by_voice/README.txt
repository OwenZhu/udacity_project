安装XGBoost Python pakage的方法（基于Ubuntu系统）

git clone --recursive https://github.com/dmlc/xgboost
cd xgboost; make -j4
cd python-package; sudo python setup.py install

大致运行时间：5分钟以内（单核CPU）