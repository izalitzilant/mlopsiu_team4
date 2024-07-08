#!/bin/bash

TAG=`cat "$PYTHONPATH/../configs/sample_tag.txt"`
SAMPLE_OUTPUT_DIR="$PYTHONPATH/../data/samples"
SAMPLE_FILENAME="$1"
SAMPLE_DVC_MSG="Added data version $TAG"

echo "$SAMPLE_OUTPUT_DIR/$SAMPLE_FILENAME"

dvc add "$SAMPLE_OUTPUT_DIR/$SAMPLE_FILENAME"

git add "$SAMPLE_OUTPUT_DIR/$SAMPLE_FILENAME.dvc"

git commit -m "$SAMPLE_DVC_MSG"
git push

# SET VERSION BEFORE DVC STAGING IN SAMPLE_DVC_VER_PATH content
git tag -a "$TAG" -m "$SAMPLE_DVC_MSG"
git push --tags



