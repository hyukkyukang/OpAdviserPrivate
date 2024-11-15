# VAETune
## Setup Dev Container
Setup dev container using .devcontainer/devcontainer.json

Fix volumes attribute in .devcontainer/docker-compose.yml to mount directories to SSDs (performances may degrade if code and /var/lib/mysql is in slow disk)
## Prepare workload
```shell
bash ./workload_preparation.sh
```
## Setup Python Environment
```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install .
```
## Run Experiments
End-to-end Comparison (OpAdviser-w/o-Optimizer (Orange Line) in Figure 7 in the OpAdviser paper)
```shell
python scripts/optimize.py --config=scripts/sysbench_rw.ini
python scripts/optimize.py --config=scripts/sysbench_wo.ini
python scripts/optimize.py --config=scripts/sysbench_ro.ini
python scripts/optimize.py --config=scripts/twitter.ini
```
### Find ground truth
```shell
mysql -ppassword -e"drop database sbrw;"
mysql -ppassword -e"create database sbrw;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbrw  \
    oltp_read_write  \
    prepare
python scripts/optimize.py --config=scripts/sysbench_rw.ini
python scripts/optimize.py --config=scripts/sysbench_rw_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database sbwrite;"
mysql -ppassword -e"create database sbwrite;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbwrite  \
    oltp_write_only  \
    prepare
python scripts/optimize.py --config=scripts/sysbench_wo.ini
python scripts/optimize.py --config=scripts/sysbench_wo_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database sbread;"
mysql -ppassword -e"create database sbread;"
sysbench  \
    --db-driver=mysql  \
    --mysql-host=localhost  \
    --mysql-port=3306  \
    --mysql-user=root  \
    --mysql-password=password  \
    --table_size=800000  \
    --tables=300  \
    --events=0  \
    --threads=80  \
    --mysql-db=sbread  \
    oltp_read_only  \
    prepare
python scripts/optimize.py --config=scripts/sysbench_ro.ini
python scripts/optimize.py --config=scripts/sysbench_ro_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database twitter;"
mysql -ppassword -e"create database twitter;"
~/jeseok/oltpbench/oltpbenchmark -b twitter -c ~/jeseok/oltpbench/config/sample_twitter_config.xml  --create=true --load=true
python scripts/optimize.py --config=scripts/twitter.ini
python scripts/optimize.py --config=scripts/twitter_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database tpcc;"
mysql -ppassword -e"create database tpcc;"
~/jeseok/oltpbench/oltpbenchmark -b tpcc -c ~/jeseok/oltpbench/config/sample_tpcc_config.xml  --create=true --load=true
python scripts/optimize.py --config=scripts/tpcc.ini
python scripts/optimize.py --config=scripts/tpcc_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database ycsb;"
mysql -ppassword -e"create database ycsb;"
~/jeseok/oltpbench/oltpbenchmark -b ycsb -c ~/jeseok/oltpbench/config/sample_ycsb_config.xml  --create=true --load=true
python scripts/optimize.py --config=scripts/ycsb.ini
python scripts/optimize.py --config=scripts/ycsb_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database wikipedia;"
mysql -ppassword -e"create database wikipedia;"
~/jeseok/oltpbench/oltpbenchmark -b wikipedia -c ~/jeseok/oltpbench/config/sample_wikipedia_config.xml  --create=true --load=true
python scripts/optimize.py --config=scripts/wikipedia.ini
python scripts/optimize.py --config=scripts/wikipedia_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database tatp;"
mysql -ppassword -e"create database tatp;"
~/jeseok/oltpbench/oltpbenchmark -b tatp -c ~/jeseok/oltpbench/config/sample_tatp_config.xml  --create=true --load=true
python scripts/optimize.py --config=scripts/tatp.ini
python scripts/optimize.py --config=scripts/tatp_ground_truth.ini
```
```shell
mysql -ppassword -e"drop database voter;"
mysql -ppassword -e"create database voter;"
~/jeseok/oltpbench/oltpbenchmark -b voter -c ~/jeseok/oltpbench/config/sample_voter_config.xml  --create=true --load=true
python scripts/optimize.py --config=scripts/voter.ini
python scripts/optimize.py --config=scripts/voter_ground_truth.ini
```
```shell
git clone https://github.com/seokjeongeum/queries-tpch-dbgen-mysql.git
cd queries-tpch-dbgen-mysql || exit
unzip TPC-H\ V3.0.1.zip
cd dbgen || exit
make
./dbgen -s 10
mysql -ppassword -e"
DROP DATABASE tpch;
CREATE DATABASE tpch;
USE tpch;

CREATE TABLE NATION  ( N_NATIONKEY  INTEGER NOT NULL,
                            N_NAME       CHAR(25) NOT NULL,
                            N_REGIONKEY  INTEGER NOT NULL,
                            N_COMMENT    VARCHAR(152));

CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
                            R_NAME       CHAR(25) NOT NULL,
                            R_COMMENT    VARCHAR(152));

CREATE TABLE PART  ( P_PARTKEY     INTEGER NOT NULL,
                          P_NAME        VARCHAR(55) NOT NULL,
                          P_MFGR        CHAR(25) NOT NULL,
                          P_BRAND       CHAR(10) NOT NULL,
                          P_TYPE        VARCHAR(25) NOT NULL,
                          P_SIZE        INTEGER NOT NULL,
                          P_CONTAINER   CHAR(10) NOT NULL,
                          P_RETAILPRICE DECIMAL(15,2) NOT NULL,
                          P_COMMENT     VARCHAR(23) NOT NULL );

CREATE TABLE SUPPLIER ( S_SUPPKEY     INTEGER NOT NULL,
                             S_NAME        CHAR(25) NOT NULL,
                             S_ADDRESS     VARCHAR(40) NOT NULL,
                             S_NATIONKEY   INTEGER NOT NULL,
                             S_PHONE       CHAR(15) NOT NULL,
                             S_ACCTBAL     DECIMAL(15,2) NOT NULL,
                             S_COMMENT     VARCHAR(101) NOT NULL);

CREATE TABLE PARTSUPP ( PS_PARTKEY     INTEGER NOT NULL,
                             PS_SUPPKEY     INTEGER NOT NULL,
                             PS_AVAILQTY    INTEGER NOT NULL,
                             PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
                             PS_COMMENT     VARCHAR(199) NOT NULL );

CREATE TABLE CUSTOMER ( C_CUSTKEY     INTEGER NOT NULL,
                             C_NAME        VARCHAR(25) NOT NULL,
                             C_ADDRESS     VARCHAR(40) NOT NULL,
                             C_NATIONKEY   INTEGER NOT NULL,
                             C_PHONE       CHAR(15) NOT NULL,
                             C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
                             C_MKTSEGMENT  CHAR(10) NOT NULL,
                             C_COMMENT     VARCHAR(117) NOT NULL);

CREATE TABLE ORDERS  ( O_ORDERKEY       INTEGER NOT NULL,
                           O_CUSTKEY        INTEGER NOT NULL,
                           O_ORDERSTATUS    CHAR(1) NOT NULL,
                           O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
                           O_ORDERDATE      DATE NOT NULL,
                           O_ORDERPRIORITY  CHAR(15) NOT NULL,
                           O_CLERK          CHAR(15) NOT NULL,
                           O_SHIPPRIORITY   INTEGER NOT NULL,
                           O_COMMENT        VARCHAR(79) NOT NULL);

CREATE TABLE LINEITEM ( L_ORDERKEY    INTEGER NOT NULL,
                             L_PARTKEY     INTEGER NOT NULL,
                             L_SUPPKEY     INTEGER NOT NULL,
                             L_LINENUMBER  INTEGER NOT NULL,
                             L_QUANTITY    DECIMAL(15,2) NOT NULL,
                             L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
                             L_DISCOUNT    DECIMAL(15,2) NOT NULL,
                             L_TAX         DECIMAL(15,2) NOT NULL,
                             L_RETURNFLAG  CHAR(1) NOT NULL,
                             L_LINESTATUS  CHAR(1) NOT NULL,
                             L_SHIPDATE    DATE NOT NULL,
                             L_COMMITDATE  DATE NOT NULL,
                             L_RECEIPTDATE DATE NOT NULL,
                             L_SHIPINSTRUCT CHAR(25) NOT NULL,
                             L_SHIPMODE     CHAR(10) NOT NULL,
                             L_COMMENT      VARCHAR(44) NOT NULL);

LOAD DATA LOCAL INFILE 'customer.tbl' INTO TABLE CUSTOMER FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'orders.tbl' INTO TABLE ORDERS FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'lineitem.tbl' INTO TABLE LINEITEM FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'nation.tbl' INTO TABLE NATION FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'partsupp.tbl' INTO TABLE PARTSUPP FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'part.tbl' INTO TABLE PART FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'region.tbl' INTO TABLE REGION FIELDS TERMINATED BY '|';
LOAD DATA LOCAL INFILE 'supplier.tbl' INTO TABLE SUPPLIER FIELDS TERMINATED BY '|';

ALTER TABLE REGION
ADD PRIMARY KEY (R_REGIONKEY);
ALTER TABLE NATION
ADD PRIMARY KEY (N_NATIONKEY);
ALTER TABLE NATION
ADD FOREIGN KEY NATION_FK1 (N_REGIONKEY) references REGION(R_REGIONKEY);
ALTER TABLE PART
ADD PRIMARY KEY (P_PARTKEY);
ALTER TABLE SUPPLIER
ADD PRIMARY KEY (S_SUPPKEY);
ALTER TABLE SUPPLIER
ADD FOREIGN KEY SUPPLIER_FK1 (S_NATIONKEY) references NATION(N_NATIONKEY);
ALTER TABLE PARTSUPP
ADD PRIMARY KEY (PS_PARTKEY,PS_SUPPKEY);
ALTER TABLE CUSTOMER
ADD PRIMARY KEY (C_CUSTKEY);
ALTER TABLE CUSTOMER
ADD FOREIGN KEY CUSTOMER_FK1 (C_NATIONKEY) references NATION(N_NATIONKEY);
ALTER TABLE LINEITEM
ADD PRIMARY KEY (L_ORDERKEY,L_LINENUMBER);
ALTER TABLE PARTSUPP
ADD FOREIGN KEY PARTSUPP_FK1 (PS_SUPPKEY) references SUPPLIER(S_SUPPKEY);
ALTER TABLE PARTSUPP
ADD FOREIGN KEY PARTSUPP_FK2 (PS_PARTKEY) references PART(P_PARTKEY);
ALTER TABLE ORDERS
ADD FOREIGN KEY ORDERS_FK1 (O_CUSTKEY) references CUSTOMER(C_CUSTKEY);
ALTER TABLE LINEITEM
ADD FOREIGN KEY LINEITEM_FK1 (L_ORDERKEY)  references ORDERS(O_ORDERKEY);
ALTER TABLE LINEITEM
ADD FOREIGN KEY LINEITEM_FK2 (L_PARTKEY,L_SUPPKEY) references PARTSUPP(PS_PARTKEY, PS_SUPPKEY);
"
python scripts/optimize.py --config=scripts/tpch.ini
python scripts/optimize.py --config=scripts/tpch_ground_truth.ini
```
```shell
chmod +x ./job.sh
bash ./job.sh
python scripts/optimize.py --config=scripts/job.ini
python scripts/optimize.py --config=scripts/job_ground_truth.ini
```
Below is original README.md
---
# OpAdviser: An Efficient Transfer Learning Based Configuration Adviser for Database Tuning

**OpAdviser** is a customized and efficient tuning system that  addresses the search space construction and the search optimizer selection  problems for database configuration tuning.



## Installation 
Installation Requirements:
- Python >= 3.6 

 ```shell
   git clone git@github.com:Blairruc-pku/OpAdviser.git && cd OpAdviser
   pip install -r requirements.txt
   pip install .
   ```




## Preparation 
####  Workload Preparation 
Please reffer to the <a href="https://github.com/Blairruc-pku/OpAdviser/blob/main/documents/workload_prepare.md" target="_blank" rel="nofollow">details instuction</a>  for preparing the workloads.
####  Database Connection Setup
To provide the database connection information, the users need to edit the `config_auto.ini`.
```ini
db = mysql
host = 127.0.0.1
port = 3306
user = root
passwd =
  ```

## Quick Start

 
1. Specify the tuning objective in `config_auto.ini`. Here are some examples.


    Performance tuning, e.g., maximizing throughputs.
    ```ini
    task_id = op1
    performance_metric = ['tps']
    ```
    
    Setup automatic space construction and optimizer recommendation
    ```ini
    ##path of data repository
    data_repo = ../repo
    ##Turn on space construction
    space_transfer = True
    only_knob = False
    only_range = False
    ##Turn on optimizer recommendation
    auto_optimizer = True
    auto_optimizer_type = learned
    ```

2. Conduct Tuning.
    ```bash
    cd scripts
    python optimize.py  --config=config_auto.ini
    ```
 

## Contact

If you have any technical questions, please submit new issues.

If you have any other questions, please contact Xinyi Zhang[zhang_xinyi@pku.edu.cn].
