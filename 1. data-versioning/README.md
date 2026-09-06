# DVC info

## we configure these below things in dvc stage command

- -n : name 
- -d : src/data_ingestion.py (data path)
- -p : params 
- -o : data/raw (output)

### command used : dvc stage add -n data_ingestion -d '1. data-versioning/src/data_ingestion.py' -o data/raw python '1. data-versioning/src/data_ingestion.py'

### command to run : dvc repro