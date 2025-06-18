#!/bin/bash

# ==============================================================================
# Loan Default Prediction API - Setup Script for MacBook
# ==============================================================================

set -e  # Exit on any error

echo "🚀 Setting up Loan Default Prediction API on MacBook..."
echo "======================================================"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"

# Create conda environment
ENV_NAME="loan-prediction"
echo "📦 Creating conda environment: $ENV_NAME"

if conda env list | grep -q "$ENV_NAME"; then
    echo "⚠️  Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=3.9 -y
echo "✅ Environment $ENV_NAME created"

# Activate environment
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install Java 17 (required for PySpark 4.0)
echo "☕ Installing OpenJDK 17..."
conda install -c conda-forge openjdk=17 -y

# Install Python packages
echo "🐍 Installing Python packages..."
conda install -c conda-forge pyspark flask -y

# Alternative: use pip for latest versions
# pip install pyspark flask

# Set up Java environment variables
echo "🔧 Setting up Java environment variables..."
JAVA_HOME_PATH="$CONDA_PREFIX/lib/jvm"

# Create directory for conda activation scripts
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"

# Create activation script
cat > "$CONDA_PREFIX/etc/conda/activate.d/java.sh" << EOF
#!/bin/bash
export JAVA_HOME=$JAVA_HOME_PATH
export PATH=\$JAVA_HOME/bin:\$PATH
echo "☕ Java environment activated: \$(java -version 2>&1 | head -n1)"
EOF

# Create deactivation script
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/java.sh" << EOF
#!/bin/bash
unset JAVA_HOME
echo "☕ Java environment deactivated"
EOF

# Make scripts executable
chmod +x "$CONDA_PREFIX/etc/conda/activate.d/java.sh"
chmod +x "$CONDA_PREFIX/etc/conda/deactivate.d/java.sh"

# Set Java environment for current session
export JAVA_HOME="$JAVA_HOME_PATH"
export PATH="$JAVA_HOME/bin:$PATH"

# Verify Java installation
echo "🔍 Verifying Java installation..."
if java -version 2>&1 | head -n1 | grep -q "openjdk"; then
    echo "✅ Java installed successfully: $(java -version 2>&1 | head -n1)"
else
    echo "❌ Java installation failed"
    exit 1
fi

# Verify PySpark installation
echo "🔍 Verifying PySpark installation..."
python -c "import pyspark; print(f'✅ PySpark {pyspark.__version__} installed successfully')"

# Verify Flask installation
echo "🔍 Verifying Flask installation..."
python -c "import flask; print(f'✅ Flask {flask.__version__} installed successfully')"

# Test PySpark with Java
echo "🧪 Testing PySpark with Java..."
python -c "
from pyspark.sql import SparkSession
try:
    spark = SparkSession.builder.appName('Test').master('local[1]').getOrCreate()
    print('✅ PySpark test successful')
    spark.stop()
except Exception as e:
    print(f'❌ PySpark test failed: {e}')
    exit(1)
"

# Create convenience scripts
echo "📝 Creating convenience scripts..."

# Create run_api.sh
cat > "run_api.sh" << 'EOF'
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
EOF

# Create test.sh
cat > "test.sh" << 'EOF'
#!/bin/bash
# Convenience script to test the Loan Prediction API

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate loan-prediction

# Set Java environment (if not already set by conda activation)
if [ -z "$JAVA_HOME" ]; then
    export JAVA_HOME=/opt/anaconda3/envs/loan-prediction/lib/jvm
    export PATH=$JAVA_HOME/bin:$PATH
fi

# Run the test
echo "🧪 Testing Loan Prediction API..."
python loan_api.py test
EOF

# Make scripts executable
chmod +x run_api.sh test.sh

echo ""
echo "🎉 Setup completed successfully!"
echo "=============================="
echo ""
echo "📋 Next steps:"
echo "1. Copy your trained model to: ../models/loan_default_model"
echo "2. Test the installation:"
echo "   conda activate loan-prediction"
echo "   ./test.sh"
echo ""
echo "3. Run the API server:"
echo "   ./run_api.sh"
echo ""
echo "4. Test the API endpoints:"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:8000/examples"
echo ""
echo "📁 Files created:"
echo "   - run_api.sh (convenience script to run API)"
echo "   - test.sh (convenience script to test API)"
echo ""
echo "⚠️  Important notes:"
echo "   - Model path: /Users/grizzlymacbookpro/Desktop/test/2025-06-16/models/loan_default_model"
echo "   - API runs on port 8000 (to avoid macOS AirPlay conflicts)"
echo "   - Java environment is automatically set when activating conda environment"
echo ""
echo "✅ Ready to deploy!"
