#!/bin/bash
# Convenience script to run the Loan Prediction API

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate loan-prediction

# Set Java environment (if not already set by conda activation)
if [ -z "$JAVA_HOME" ]; then
    export JAVA_HOME=/opt/anaconda3/envs/loan-prediction/lib/jvm
    export PATH=$JAVA_HOME/bin:$PATH
fi

# Run the API
echo "🚀 Starting Loan Prediction API..."
python loan_api.py
