#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loan Default Prediction API
รัน Python + PySpark ใน Conda environment บน MacBook
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from flask import Flask, request, jsonify

class MacBookLoanPredictor:
    """
    Loan Default Predictor สำหรับ MacBook ใช้ Conda + PySpark
    """
    
    def __init__(self):
        self.spark = None
        self.model = None
        self.model_path = "/Users/grizzlymacbookpro/Desktop/test/2025-06-16/models/loan_default_model"
        self.initialize()
    
    def initialize(self):
        """
        เริ่มต้น Spark และโหลด model
        """
        print("🚀 Initializing MacBook Loan Predictor...")
        self.setup_spark()
        self.load_model()
        print("✅ Initialization completed!")
    
    def setup_spark(self):
        """
        ตั้งค่า Spark สำหรับ MacBook
        """
        try:
            self.spark = SparkSession.builder \
                .appName("MacBookLoanAPI") \
                .master("local[*]") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "1g") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .getOrCreate()
            
            # ลด log level
            self.spark.sparkContext.setLogLevel("ERROR")
            
            print(f"✅ Spark {self.spark.version} initialized on MacBook")
            
        except Exception as e:
            print(f"❌ Spark initialization failed: {e}")
            raise e
    
    def load_model(self):
        """
        โหลด model จาก MacBook path
        """
        try:
            if os.path.exists(self.model_path):
                self.model = PipelineModel.load(self.model_path)
                print(f"✅ Model loaded from: {self.model_path}")
                print(f"✅ Pipeline stages: {len(self.model.stages)}")
            else:
                print(f"❌ Model not found at: {self.model_path}")
                print("Available files in parent directory:")
                parent_dir = os.path.dirname(self.model_path)
                if os.path.exists(parent_dir):
                    for item in os.listdir(parent_dir):
                        print(f"  - {item}")
                raise FileNotFoundError(f"Model not found: {self.model_path}")
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
    
    def create_schema(self):
        """
        สร้าง schema สำหรับ input data
        """
        return StructType([
            StructField("int_rate", FloatType(), True),
            StructField("installment", FloatType(), True),
            StructField("pub_rec", FloatType(), True),
            StructField("loan_amnt", FloatType(), True),
            StructField("annual_inc", FloatType(), True),
            StructField("term", IntegerType(), True),
            StructField("grade", StringType(), True),
            StructField("home_ownership", StringType(), True),
            StructField("verification_status", StringType(), True),
            StructField("addr_state", StringType(), True)
        ])
    
    def validate_input(self, data):
        """
        ตรวจสอบ input data
        """
        required_fields = [
            'int_rate', 'installment', 'pub_rec', 'loan_amnt', 'annual_inc', 'term',
            'grade', 'home_ownership', 'verification_status', 'addr_state'
        ]
        
        missing = [field for field in required_fields if field not in data]
        if missing:
            return False, f"Missing fields: {missing}"
        
        try:
            # ตรวจสอบ data types
            float(data['int_rate'])
            float(data['installment'])
            float(data['pub_rec'])
            float(data['loan_amnt'])
            float(data['annual_inc'])
            int(data['term'])
            
            # ตรวจสอบ string fields
            for field in ['grade', 'home_ownership', 'verification_status', 'addr_state']:
                if not isinstance(data[field], str) or len(data[field]) == 0:
                    return False, f"Invalid {field}: must be non-empty string"
            
            return True, "Valid"
            
        except (ValueError, TypeError) as e:
            return False, f"Data type error: {e}"
    
    def predict(self, loan_data):
        """
        ทำนาย loan default
        """
        try:
            # Validate input
            is_valid, message = self.validate_input(loan_data)
            if not is_valid:
                return {"error": message, "status": "invalid_input"}
            
            # สร้าง DataFrame
            schema = self.create_schema()
            df = self.spark.createDataFrame([loan_data], schema)
            
            # Predict
            predictions = self.model.transform(df)
            
            # ดึงผลลัพธ์
            result = predictions.collect()[0]
            prediction_code = result['prediction']
            probability = result['probability']
            
            # แปลง probability vector
            prob_array = probability.toArray().tolist()
            prob_fully_paid = prob_array[0]
            prob_charged_off = prob_array[1]
            
            # กำหนด label
            prediction_label = "Charged Off" if prediction_code == 1.0 else "Fully Paid"
            
            # คำนวณ DTI ratio
            dti_ratio = loan_data['loan_amnt'] / loan_data['annual_inc'] if loan_data['annual_inc'] > 0 else 0
            
            return {
                "prediction_code": int(prediction_code),
                "prediction_label": prediction_label,
                "probability_fully_paid": round(prob_fully_paid, 4),
                "probability_charged_off": round(prob_charged_off, 4),
                "confidence": round(max(prob_array), 4),
                "calculated_dti": round(dti_ratio, 4),
                "input_data": loan_data,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "input_data": loan_data,
                "status": "prediction_failed"
            }
    
    def predict_batch(self, loan_list):
        """
        ทำนายหลายรายการ
        """
        try:
            # Validate all inputs
            valid_loans = []
            errors = []
            
            for i, loan in enumerate(loan_list):
                is_valid, message = self.validate_input(loan)
                if is_valid:
                    valid_loans.append(loan)
                else:
                    errors.append({"index": i, "error": message, "data": loan})
            
            if not valid_loans:
                return {
                    "predictions": [],
                    "validation_errors": errors,
                    "status": "all_invalid"
                }
            
            # สร้าง DataFrame
            schema = self.create_schema()
            df = self.spark.createDataFrame(valid_loans, schema)
            
            # Predict
            predictions = self.model.transform(df)
            results = predictions.collect()
            
            # ประมวลผลลัพธ์
            prediction_results = []
            for i, (row, original_data) in enumerate(zip(results, valid_loans)):
                prediction_code = row['prediction']
                probability = row['probability']
                prob_array = probability.toArray().tolist()
                
                prediction_label = "Charged Off" if prediction_code == 1.0 else "Fully Paid"
                dti_ratio = original_data['loan_amnt'] / original_data['annual_inc'] if original_data['annual_inc'] > 0 else 0
                
                prediction_results.append({
                    "index": i,
                    "prediction_code": int(prediction_code),
                    "prediction_label": prediction_label,
                    "probability_fully_paid": round(prob_array[0], 4),
                    "probability_charged_off": round(prob_array[1], 4),
                    "confidence": round(max(prob_array), 4),
                    "calculated_dti": round(dti_ratio, 4),
                    "input_data": original_data
                })
            
            return {
                "predictions": prediction_results,
                "validation_errors": errors,
                "total_input": len(loan_list),
                "valid_predictions": len(prediction_results),
                "invalid_inputs": len(errors),
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "batch_failed"
            }
    
    def get_system_info(self):
        """
        ข้อมูลระบบ
        """
        return {
            "platform": "MacBook",
            "python_version": sys.version,
            "spark_version": self.spark.version if self.spark else None,
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        }
    
    def shutdown(self):
        """
        ปิด Spark session
        """
        if self.spark:
            self.spark.stop()
            print("✅ Spark session stopped")

# Flask API
app = Flask(__name__)
predictor = None

def initialize_predictor():
    """
    เริ่มต้น predictor
    """
    global predictor
    if predictor is None:
        try:
            predictor = MacBookLoanPredictor()
        except Exception as e:
            print(f"❌ Failed to initialize predictor: {e}")
            raise e

@app.route('/health', methods=['GET'])
def health():
    """
    Health check
    """
    if predictor is None:
        initialize_predictor()
    
    if predictor and predictor.spark and predictor.model:
        return jsonify({
            "status": "healthy",
            "service": "macbook-loan-prediction",
            "spark_running": True,
            "model_loaded": True
        })
    else:
        return jsonify({
            "status": "unhealthy",
            "service": "macbook-loan-prediction",
            "spark_running": predictor.spark is not None if predictor else False,
            "model_loaded": predictor.model is not None if predictor else False
        }), 500

@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Single prediction
    """
    try:
        if predictor is None:
            initialize_predictor()
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        result = predictor.predict(data)
        
        if result.get("status") != "success":
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction
    """
    try:
        if predictor is None:
            initialize_predictor()
        
        data = request.get_json()
        if not data or 'loans' not in data:
            return jsonify({"error": "Expected JSON with 'loans' array"}), 400
        
        result = predictor.predict_batch(data['loans'])
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def system_info():
    """
    System information
    """
    if predictor is None:
        initialize_predictor()
    
    return jsonify(predictor.get_system_info())

@app.route('/examples', methods=['GET'])
def examples():
    """
    ตัวอย่างข้อมูลสำหรับทดสอบ
    """
    return jsonify({
        "low_risk": {
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
        },
        "medium_risk": {
            "int_rate": 15.5,
            "installment": 750.0,
            "pub_rec": 1.0,
            "loan_amnt": 25000.0,
            "annual_inc": 65000.0,
            "term": 60,
            "grade": "C",
            "home_ownership": "RENT",
            "verification_status": "Source Verified",
            "addr_state": "TX"
        },
        "high_risk": {
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
        }
    })

def test_local():
    """
    ทดสอบ local
    """
    print("🧪 Testing MacBook PySpark API...")
    
    try:
        # Test predictor
        test_predictor = MacBookLoanPredictor()
        
        # Test data
        test_loan = {
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
        
        # Single prediction
        result = test_predictor.predict(test_loan)
        if result.get("status") == "success":
            print(f"✅ Single prediction: {result['prediction_label']} (confidence: {result['confidence']})")
        else:
            print(f"❌ Single prediction failed: {result.get('error')}")
        
        # Cleanup
        test_predictor.shutdown()
        print("✅ Local test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_local()
    else:
        print("🚀 Starting MacBook Loan Prediction API")
        print("📱 Platform: macOS")
        print("🐍 Environment: Conda")
        print("⚡ Engine: PySpark")
        print("=" * 50)
        
        # Initialize predictor before starting Flask
        initialize_predictor()
        
        app.run(host='0.0.0.0', port=8000, debug=False)
