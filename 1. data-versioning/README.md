# DVC info

## we configure these below things in dvc stage command

- -n : name 
- -d : src/data_ingestion.py (data path)
- -p : params 
- -o : data/raw (output)

### command used : dvc stage add -n data_ingestion -d src/data_ingestion.py -o data/raw python '1. data-versioning/src/data_ingestion.py'

### command to run : dvc repro

## actual command if there are multiple dependencies : 


2. dvc stage add -n data_preprocessing -d "1. data-versioning/src/pre_process.py" -d data/raw -o data/processed python "1. data-versioning/src/pre_process.py"

3. dvc stage add --force -n feature_engineering -d "1. data-versioning/src/feature_engineering.py" -d data/processed -o data/features python "1. data-versioning/src/feature_engineering.py"

4. 