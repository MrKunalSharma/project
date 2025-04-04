from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

app = FastAPI(title="Network Optimizer API")

# Initialize model and scalers as None
model = None
scaler_X = None
scaler_y = None

class NetworkConditions(BaseModel):
    bandwidth: float
    throughput: float
    packet_loss: float
    latency: float
    jitter: float

def load_model():
    global model, scaler_X, scaler_y
    try:
        # Load model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Initialize scalers
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        
        # Set some default scaling parameters
        scaler_X.scale_ = np.array([0.1, 0.1, 0.01, 0.01, 0.01])
        scaler_X.min_ = np.array([0, 0, 0, 0, 0])
        scaler_y.scale_ = np.array([0.01])
        scaler_y.min_ = np.array([0])
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def get_video_recommendation(bandwidth: float, conditions: Dict[str, float]) -> Dict[str, Any]:
    quality_levels = {
        'low': {'resolution': '480p', 'fps': 24, 'bitrate_factor': 0.6},
        'medium': {'resolution': '720p', 'fps': 30, 'bitrate_factor': 0.7},
        'high': {'resolution': '1080p', 'fps': 30, 'bitrate_factor': 0.8}
    }
    
    network_score = 1.0
    network_score -= conditions['packet_loss'] * 0.05
    network_score -= (conditions['latency'] / 100)
    network_score -= conditions['jitter'] * 0.1
    network_score = max(0, min(1, network_score))
    
    if network_score < 0.3 or bandwidth < 1.5:
        quality = quality_levels['low']
    elif network_score < 0.7 or bandwidth < 2.5:
        quality = quality_levels['medium']
    else:
        quality = quality_levels['high']
    
    actual_bitrate = bandwidth * quality['bitrate_factor'] * network_score
    
    return {
        'resolution': quality['resolution'],
        'fps': quality['fps'],
        'bitrate': f"{actual_bitrate:.1f} Mbps",
        'network_quality_score': f"{network_score:.2f}",
        'adaptive_streaming': network_score < 0.8
    }

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def read_root():
    return {
        "message": "Network Optimizer API",
        "version": "1.0",
        "endpoints": [
            "/predict - POST: Predict network congestion",
            "/optimize - POST: Optimize network settings"
        ]
    }

@app.post("/predict/")
async def predict_network(conditions: NetworkConditions):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        input_data = np.array([[
            conditions.bandwidth,
            conditions.throughput,
            conditions.packet_loss,
            conditions.latency,
            conditions.jitter
        ]])
        
        input_scaled = scaler_X.transform(input_data)
        pred_scaled = model.predict(input_scaled, verbose=0)
        prediction = scaler_y.inverse_transform(pred_scaled)[0][0]
        prediction = max(0, min(100, prediction))
        
        video_rec = get_video_recommendation(conditions.bandwidth, conditions.dict())
        
        return {
            "status": "success",
            "congestion_prediction": float(prediction),
            "video_recommendations": video_rec,
            "network_conditions": conditions.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/")
async def optimize_network(conditions: NetworkConditions):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        min_bw = 0.5
        max_bw = 10.0
        steps = 50
        
        # Adjust max_bw based on conditions
        if conditions.packet_loss > 20:
            max_bw = min(max_bw, 1.0)
        elif conditions.packet_loss > 10:
            max_bw = min(max_bw, 2.0)
        elif conditions.packet_loss > 5:
            max_bw = min(max_bw, 3.0)

        bandwidths = np.linspace(min_bw, max_bw, steps)
        results = []
        
        for bw in bandwidths:
            test_conditions = conditions.dict()
            test_conditions['bandwidth'] = bw
            
            input_data = np.array([[
                test_conditions['bandwidth'],
                test_conditions['throughput'],
                test_conditions['packet_loss'],
                test_conditions['latency'],
                test_conditions['jitter']
            ]])
            
            input_scaled = scaler_X.transform(input_data)
            pred_scaled = model.predict(input_scaled, verbose=0)
            congestion = float(scaler_y.inverse_transform(pred_scaled)[0][0])
            
            results.append({
                'bandwidth': bw,
                'congestion': congestion
            })
        
        optimal = min(results, key=lambda x: x['congestion'])
        video_rec = get_video_recommendation(optimal['bandwidth'], conditions.dict())
        
        return {
            "status": "success",
            "current_conditions": conditions.dict(),
            "optimal_bandwidth": optimal['bandwidth'],
            "expected_congestion": optimal['congestion'],
            "video_recommendations": video_rec
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
