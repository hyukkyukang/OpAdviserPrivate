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
