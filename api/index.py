from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Network Optimizer API")

class NetworkConditions(BaseModel):
    bandwidth: float
    throughput: float
    packet_loss: float
    latency: float
    jitter: float

@app.get("/")
def read_root():
    return {
        "message": "Network Optimizer API",
        "version": "1.0",
        "status": "running"
    }

@app.post("/predict/")
async def predict_network(conditions: NetworkConditions):
    try:
        # Simple prediction logic
        score = 100.0
        score -= conditions.packet_loss * 2
        score -= conditions.latency * 0.5
        score -= conditions.jitter * 5
        score += conditions.throughput * 10
        score = max(0, min(100, score))
        
        return {
            "status": "success",
            "congestion_prediction": float(score),
            "network_conditions": conditions.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/")
async def optimize_network(conditions: NetworkConditions):
    try:
        min_bw = 0.5
        max_bw = 10.0
        steps = 20
        
        bandwidths = np.linspace(min_bw, max_bw, steps)
        results = []
        
        for bw in bandwidths:
            score = 100.0
            score -= conditions.packet_loss * 2
            score -= conditions.latency * 0.5
            score -= conditions.jitter * 5
            score += (bw * conditions.throughput) * 10
            score = max(0, min(100, score))
            
            results.append({
                'bandwidth': float(bw),
                'predicted_performance': float(score)
            })
        
        optimal = max(results, key=lambda x: x['predicted_performance'])
        
        return {
            "status": "success",
            "current_conditions": conditions.dict(),
            "optimization_results": results,
            "optimal_settings": optimal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
