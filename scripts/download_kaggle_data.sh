#!/usr/bin/env bash
# Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables
kaggle datasets download -d <dataset-owner>/<dataset-name> -p ./data --unzip