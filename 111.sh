#!/bin/bash
db="read"
rm -rf sysbench
git clone https://github.com/akopytov/sysbench.git && \
    cd sysbench && \
    git checkout ead2689ac6f61c5e7ba7c6e19198b86bd3a51d3c && \
    ./autogen.sh && \
    ./configure && \
    make && make install
mysql -ppassword -e"drop database sb${db};"
mysql -ppassword -e"create database sb${db};"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3307  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sb${db}  \
    oltp_${db}_only  \
    prepare
cd ~/OpAdviserPrivate
update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
python -m pip install --upgrade pip
pip install --user --upgrade setuptools
pip install --upgrade wheel
python -m pip install -r requirements.txt
python -m pip install .
for optimize_method in "DDPG" "GA" "MBO" "SMAC"; do
  lowercase="${optimize_method,,}"
  for knob_num in 118 29 12; do
    python3 scripts/optimize.py \
    --config=scripts/cluster.ini \
    --knob_config_file=scripts/experiment/gen_knobs/SYSBENCH_randomforest.json \
    --knob_num=$knob_num \
    --dbname=sb${db} \
    --workload=sysbench \
    --task_id="sysbench_${lowercase}_${knob_num}" \
    --optimize_method="$optimize_method"
  done
done
