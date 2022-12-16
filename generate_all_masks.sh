#!/bin/bash

python3 generate_mask.py --test-dataset Figaro1k --model models/unet/Figaro1k-no-aug/model-2022-12-16_12\:03\:12/model.h5
python3 generate_mask.py --test-dataset Figaro1k --model models/unet/Lfw-aug/model-2022-12-16_12\:06\:21/model.h5
python3 generate_mask.py --test-dataset Figaro1k --model models/unet/Lfw-no-aug/model-2022-12-16_12\:13\:40/model.h5
python3 generate_mask.py --test-dataset Figaro1k --model models/unet/Lfw+Figaro1k-no-aug/model-2022-12-16_12\:30\:28/model.h5
python3 generate_mask.py --test-dataset Figaro1k --model models/unet/Lfw+Figaro1k-aug/model-2022-12-16_12\:20\:17/model.h5

python3 generate_mask.py --test-dataset Lfw --model models/unet/Figaro1k-no-aug/model-2022-12-16_12\:03\:12/model.h5
python3 generate_mask.py --test-dataset Lfw --model models/unet/Lfw-aug/model-2022-12-16_12\:06\:21/model.h5
python3 generate_mask.py --test-dataset Lfw --model models/unet/Lfw-no-aug/model-2022-12-16_12\:13\:40/model.h5
python3 generate_mask.py --test-dataset Lfw --model models/unet/Lfw+Figaro1k-no-aug/model-2022-12-16_12\:30\:28/model.h5
python3 generate_mask.py --test-dataset Lfw --model models/unet/Lfw+Figaro1k-aug/model-2022-12-16_12\:20\:17/model.h5

python3 generate_mask.py --test-dataset Lfw+Figaro1k --model models/unet/Figaro1k-no-aug/model-2022-12-16_12\:03\:12/model.h5
python3 generate_mask.py --test-dataset Lfw+Figaro1k --model models/unet/Lfw-aug/model-2022-12-16_12\:06\:21/model.h5
python3 generate_mask.py --test-dataset Lfw+Figaro1k --model models/unet/Lfw-no-aug/model-2022-12-16_12\:13\:40/model.h5
python3 generate_mask.py --test-dataset Lfw+Figaro1k --model models/unet/Lfw+Figaro1k-no-aug/model-2022-12-16_12\:30\:28/model.h5
python3 generate_mask.py --test-dataset Lfw+Figaro1k --model models/unet/Lfw+Figaro1k-aug/model-2022-12-16_12\:20\:17/model.h5