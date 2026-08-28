#!/bin/bash

python -m data_generation.examples.generate_elastic_datasets --config=data_generation/configs/elastic_params.yaml

python -m data_generation.examples.generate_noisy_elastic_datasets --config=data_generation/configs/noise_params.yaml

python -m mechanics.examples.run_elastic_exp --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/elastic_exp.yaml

python -m mechanics.examples.run_elastic_noise_reg --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/reg_exp.yaml

python -m mechanics.examples.run_elastic_noise --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/noise_exp.yaml

python -m mechanics.examples.run_micro_image_exp --config=mechanics/configs/optical_flow.yaml --config=mechanics/configs/general.yaml --config=mechanics/configs/micro_exp.yaml