#!/bin/bash

echo `pwd`
cd "$PYTHONPATH/../.."
echo `pwd`
TAG=`cat "$PYTHONPATH/../configs/sample_tag.txt"`
echo $TAG