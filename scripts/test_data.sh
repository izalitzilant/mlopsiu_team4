#!/bin/bash

# install
python3 ./src/data.py

# validate
python3 ./src/validate_data.py

if [[ $? -eq 0 ]]; then
    echo "sample is valid"

    dvc add ./data/samples/sample.csv
    dvc commit -m "sample.csv is proven valid "
    dvc push
else
    echo "sample is not valid"
fi

