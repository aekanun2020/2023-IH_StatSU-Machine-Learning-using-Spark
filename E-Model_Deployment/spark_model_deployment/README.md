# Loan Default Prediction API

## 📋 Overview

A machine learning API for predicting loan default risk using PySpark and Flask. The model predicts whether a loan applicant will fully pay their loan or default (charge off).

## 🏗️ Architecture

- **Machine Learning**: PySpark ML Pipeline with Decision Tree Classifier
- **API Framework**: Flask REST API
- **Environment**: Conda + Python 3.9
- **Platform**: macOS (Apple Silicon/Intel)
- **Java Runtime**: OpenJDK 17

## 📁 Project Structure

```
spark_model_deployment/
├── loan_api.py          # Main API application
├── setup.sh             # Installation script
├── run_api.sh           # Convenience script to run API
├── test.sh              # Convenience script to test API
├── README.md            # This documentation
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### 1. Prerequisites

- macOS (Apple Silicon or Intel)
- Conda/Miniconda installed
- Trained PySpark model at: `/Users/grizzlymacbookpro/Desktop/test/2025-06-16/models/loan_default_model`

### 2. Installation

```bash
# Make setup script executable
chmod +x setup.sh

# Run installation
./setup.sh
```

### 3. Testing

```bash
# Test the installation
./test.sh
```

### 4. Running the API

```bash
# Start the API server
./run_api.sh
```

The API will be available at `http://localhost:8000`

## 🔧 Manual Setup (Alternative)

If you prefer manual setup:

### Step 1: Create Conda Environment

```bash
# Create environment
conda create -n loan-prediction python=3.9 -y
conda activate loan-prediction

# Install Java 17
conda install -c conda-forge openjdk=17 -y

# Install Python packages
conda install -c conda-forge pyspark flask -y
```

### Step 2: Set Java Environment

```bash
# Set Java path
export JAVA_HOME=/opt/anaconda3/envs/loan-prediction/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH

# Verify Java
java -version
```

### Step 3: Test PySpark

```bash
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Test').master('local[1]').getOrCreate()
print('PySpark working!')
spark.stop()
"
```

## 🌐 API Endpoints

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "macbook-loan-prediction",
  "spark_running": true,
  "model_loaded": true
}
```

### System Information
```bash
GET /info
```

**Response:**
```json
{
  "platform": "MacBook",
  "python_version": "3.9.x",
  "spark_version": "4.0.0",
  "model_path": "/Users/.../models/loan_default_model",
  "model_loaded": true,
  "conda_env": "loan-prediction"
}
```

### Sample Data
```bash
GET /examples
```

**Response:**
```json
{
  "low_risk": { ... },
  "medium_risk": { ... },
  "high_risk": { ... }
}
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "int_rate": 15.5,
  "installment": 750.0,
  "pub_rec": 0.0,
  "loan_amnt": 25000.0,
  "annual_inc": 75000.0,
  "term": 36,
  "grade": "B",
  "home_ownership": "RENT",
  "verification_status": "Verified",
  "addr_state": "CA"
}
```

**Response:**
```json
{
  "prediction_code": 1,
  "prediction_label": "Charged Off",
  "probability_fully_paid": 0.4966,
  "probability_charged_off": 0.5034,
  "confidence": 0.5034,
  "calculated_dti": 0.3333,
  "input_data": { ... },
  "status": "success"
}
```

### Batch Prediction
```bash
POST /predict_batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "loans": [
    { 
      "int_rate": 8.5,
      "installment": 450.0,
      ...
    },
    {
      "int_rate": 25.0,
      "installment": 1500.0,
      ...
    }
  ]
}
```

## 🧪 Testing Results

### Test Case 1: Low Risk Customer
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "int_rate": 8.5,
    "installment": 450.0,
    "pub_rec": 0.0,
    "loan_amnt": 15000.0,
    "annual_inc": 85000.0,
    "term": 36,
    "grade": "A",
    "home_ownership": "OWN",
    "verification_status": "Verified",
    "addr_state": "CA"
  }'
```

**Result:**
- Prediction: **Fully Paid**
- Confidence: **69.17%**
- DTI: **17.65%**

### Test Case 2: Medium Risk Customer
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "int_rate": 15.5,
    "installment": 750.0,
    "pub_rec": 0.0,
    "loan_amnt": 25000.0,
    "annual_inc": 75000.0,
    "term": 36,
    "grade": "B",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "addr_state": "CA"
  }'
```

**Result:**
- Prediction: **Charged Off**
- Confidence: **50.34%**
- DTI: **33.33%**

### Test Case 3: High Risk Customer
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "int_rate": 25.0,
    "installment": 1500.0,
    "pub_rec": 2.0,
    "loan_amnt": 35000.0,
    "annual_inc": 35000.0,
    "term": 60,
    "grade": "E",
    "home_ownership": "RENT",
    "verification_status": "Not Verified",
    "addr_state": "FL"
  }'
```

**Result:**
- Prediction: **Charged Off**
- Confidence: **93.44%**
- DTI: **100%** (⚠️ Very High Risk!)

## 📊 Model Performance Summary

| Risk Level | Grade | Interest Rate | DTI | Prediction | Confidence |
|------------|-------|---------------|-----|------------|------------|
| Low | A | 8.5% | 17.65% | Fully Paid | 69.17% |
| Medium | B | 15.5% | 33.33% | Charged Off | 50.34% |
| High | E | 25.0% | 100% | Charged Off | 93.44% |

## 🔍 Model Features

### Input Features
- **int_rate**: Interest rate (%)
- **installment**: Monthly payment amount ($)
- **pub_rec**: Number of public records
- **loan_amnt**: Loan amount ($)
- **annual_inc**: Annual income ($)
- **term**: Loan term (months)
- **grade**: Loan grade (A-G)
- **home_ownership**: Home ownership status
- **verification_status**: Income verification status
- **addr_state**: State code

### Output
- **prediction_code**: 0 = Fully Paid, 1 = Charged Off
- **prediction_label**: Human-readable prediction
- **probability_fully_paid**: Probability of paying in full
- **probability_charged_off**: Probability of defaulting
- **confidence**: Model confidence (max probability)
- **calculated_dti**: Debt-to-Income ratio

## 🔧 Configuration

### Model Path
Update the model path in `loan_api.py`:
```python
self.model_path = "/Users/grizzlymacbookpro/Desktop/test/2025-06-16/models/loan_default_model"
```

### Port Configuration
The API runs on port 8000 by default (to avoid macOS AirPlay conflicts):
```python
app.run(host='0.0.0.0', port=8000, debug=False)
```

### Spark Configuration
Spark settings in `MacBookLoanPredictor.setup_spark()`:
```python
.config("spark.executor.memory", "2g")
.config("spark.driver.memory", "1g")
.config("spark.sql.adaptive.enabled", "true")
```

## 🐛 Troubleshooting

### Java Issues
```bash
# Check Java version
java -version

# Set Java path
export JAVA_HOME=/opt/anaconda3/envs/loan-prediction/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
```

### PySpark Issues
```bash
# Test PySpark installation
python -c "import pyspark; print(pyspark.__version__)"

# Test Spark session
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Test').master('local[1]').getOrCreate()
print('Success!')
spark.stop()
"
```

### Port Conflicts
If port 8000 is in use:
```bash
# Check what's using the port
lsof -i :8000

# Use different port
python -c "
exec(open('loan_api.py').read().replace('port=8000', 'port=9000'))
"
```

### Model Loading Issues
- Ensure model exists at specified path
- Check file permissions
- Verify model was saved with compatible PySpark version

## 📈 Performance Metrics

### System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Multi-core recommended for better Spark performance
- **Storage**: 1GB for environment + model size

### Response Times
- **Single prediction**: ~100-500ms
- **Batch prediction**: ~200ms-2s (depends on batch size)
- **Model loading**: ~5-10s (one-time startup cost)

## 🚀 Production Deployment

### Using Gunicorn (Recommended)
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn --bind 0.0.0.0:8000 --workers 1 --timeout 120 loan_api:app
```

### Docker Deployment
```dockerfile
FROM conda/miniconda3

WORKDIR /app
COPY . .

RUN ./setup.sh
CMD ["./run_api.sh"]
```

### Load Balancing
- Use multiple Gunicorn workers
- Consider nginx as reverse proxy
- Monitor memory usage per worker

## 📝 API Usage Examples

### Python Client
```python
import requests

# Single prediction
response = requests.post('http://localhost:8000/predict', json={
    "int_rate": 15.5,
    "installment": 750.0,
    "pub_rec": 0.0,
    "loan_amnt": 25000.0,
    "annual_inc": 75000.0,
    "term": 36,
    "grade": "B",
    "home_ownership": "RENT",
    "verification_status": "Verified",
    "addr_state": "CA"
})

result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript Client
```javascript
fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        int_rate: 15.5,
        installment: 750.0,
        pub_rec: 0.0,
        loan_amnt: 25000.0,
        annual_inc: 75000.0,
        term: 36,
        grade: "B",
        home_ownership: "RENT",
        verification_status: "Verified",
        addr_state: "CA"
    })
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction_label);
    console.log('Confidence:', data.confidence);
});
```

## 🔒 Security Considerations

### Input Validation
- All inputs are validated for type and range
- SQL injection protection through parameterized queries
- Input sanitization for string fields

### API Security
- Consider adding authentication (API keys, JWT)
- Rate limiting for production use
- HTTPS in production environment
- Input size limits

### Model Security
- Model files should be read-only
- Secure model storage and access
- Model versioning and rollback capabilities

## 📊 Monitoring and Logging

### Application Logs
- Spark logs are set to ERROR level
- API access logs via Flask/Gunicorn
- Custom application logging available

### Health Monitoring
- `/health` endpoint for load balancer checks
- System resource monitoring recommended
- Model performance monitoring

### Metrics to Track
- Response times
- Prediction accuracy over time
- Resource utilization
- Error rates

## 🔄 Model Updates

### Updating the Model
1. Train new model with same feature schema
2. Save model to same path with backup
3. Restart API service
4. Verify with test endpoints

### A/B Testing
- Deploy multiple model versions
- Route traffic based on criteria
- Compare performance metrics

## 📞 Support

### Common Issues
1. **Java not found**: Ensure OpenJDK 17 is installed
2. **Port conflicts**: Use different port or disable AirPlay
3. **Model not found**: Check model path and permissions
4. **Memory issues**: Adjust Spark memory settings

### Getting Help
- Check logs for detailed error messages
- Verify environment activation
- Test individual components separately

## 📜 License

This project is for educational and demonstration purposes.

## 🎉 Success Metrics

✅ **Model Deployment**: Successfully deployed PySpark ML model  
✅ **API Performance**: Sub-second response times  
✅ **Accuracy**: Model predictions align with risk profiles  
✅ **Reliability**: Robust error handling and validation  
✅ **Scalability**: Ready for production workloads  

---

**🚀 Ready for Production Deployment!**
