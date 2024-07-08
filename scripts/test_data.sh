#!/bin/bash

# install
python3 ./src/data.py

# validate
python3 ./src/validate_data.py

if [[ $? -eq 0 ]]; then
    echo "sample is valid"

    bash ./scripts/save_sample.sh
    dvc push
else
    echo "sample is not valid"
fi

