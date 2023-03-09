#!/bin/bash

# Set up paths and parameters
DATASET_NAME="weissmann_rpe1"
OUTPUT_DIRECTORY="/path/to/output/"
DATA_DIRECTORY="/path/to/data/storage"
TRAINING_REGIME="partial_interventional"
PARTIAL_INTERVENTION_SEED=0
MODEL_NAME="custom"
INFERENCE_FUNCTION_FILE_PATH="/path/to/custom_inference_function.py"
SUBSET_DATA=1.0
MODEL_SEED=0
PLOT_DIRECTORY="/path/to/plots"

# Loop over the different fraction values
for FRACTION in 0.25 0.5 0.75 1.0; do
    # Construct the command
    COMMAND="causalbench_run \
        --dataset_name ${DATASET_NAME} \
        --output_directory ${OUTPUT_DIRECTORY} \
        --data_directory ${DATA_DIRECTORY} \
        --training_regime ${TRAINING_REGIME} \
        --partial_intervention_seed ${PARTIAL_INTERVENTION_SEED} \
        --fraction_partial_intervention ${FRACTION} \
        --model_name ${MODEL_NAME} \
        --inference_function_file_path ${INFERENCE_FUNCTION_FILE_PATH} \
        --subset_data ${SUBSET_DATA} \
        --model_seed ${MODEL_SEED} \
        --do_filter"

    # Run the command
    echo "Running command: ${COMMAND}"
    eval ${COMMAND}
done

# Generate plots
PLOTS_SCRIPT="python scripty/plots.py ${PLOT_DIRECTORY}  ${OUTPUT_DIRECTORY}"
echo "Running plots script: ${PLOTS_SCRIPT}"
eval ${PLOTS_SCRIPT}
