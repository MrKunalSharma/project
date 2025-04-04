from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
import json

app = FastAPI(
    title="Network Performance Optimizer",
    description="Final Year Project - Advanced Network Performance Optimization System",
    version="2.0"
)

class NetworkConditions(BaseModel):
    bandwidth: float = Field(..., gt=0, description="Network bandwidth in Mbps")
    throughput: float = Field(..., ge=0, le=1, description="Network throughput ratio")
    packet_loss: float = Field(..., ge=0, le=100, description="Packet loss percentage")
    latency: float = Field(..., ge=0, description="Network latency in ms")
    jitter: float = Field(..., ge=0, description="Network jitter in ms")

class NetworkAnalysis(BaseModel):
    network_score: float
    congestion_level: str
    performance_rating: str
    recommendations: List[str]

def analyze_network(conditions: NetworkConditions) -> NetworkAnalysis:
    # Calculate network score (0-100)
    score = 100.0
    score -= conditions.packet_loss * 2  # Packet loss impact
    score -= conditions.latency * 0.5    # Latency impact
    score -= conditions.jitter * 5       # Jitter impact
    score += conditions.throughput * 20  # Throughput positive impact
    score += min(conditions.bandwidth * 5, 30)  # Bandwidth positive impact (capped)
    
    # Normalize score
    score = max(0, min(100, score))
    
    # Determine congestion level
    if score >= 80:
        congestion = "Low"
    elif score >= 60:
        congestion = "Moderate"
    elif score >= 40:
        congestion = "High"
    else:
        congestion = "Severe"
    
    # Performance rating
    if score >= 90:
        rating = "Excellent"
    elif score >= 75:
        rating = "Good"
    elif score >= 60:
        rating = "Fair"
    elif score >= 40:
        rating = "Poor"
    else:
        rating = "Critical"
    
    # Generate recommendations
    recommendations = []
    if conditions.packet_loss > 5:
        recommendations.append("High packet loss detected. Consider checking network hardware or ISP service.")
    if conditions.latency > 100:
        recommendations.append("High latency detected. Consider using a closer server or check network routing.")
    if conditions.jitter > 30:
        recommendations.append("High jitter detected. QoS settings adjustment recommended.")
    if conditions.throughput < 0.3:
        recommendations.append("Low throughput detected. Bandwidth upgrade or traffic optimization recommended.")
    
    return NetworkAnalysis(
        network_score=score,
        congestion_level=congestion,
        performance_rating=rating,
        recommendations=recommendations
    )

def get_video_recommendations(conditions: NetworkConditions, score: float) -> Dict:
    if score >= 80:
        return {
            "resolution": "4K (2160p)",
            "fps": 60,
            "bitrate": f"{min(conditions.bandwidth * 0.8, 35):.1f} Mbps",
            "codec": "H.265/HEVC",
            "streaming_quality": "Ultra High"
        }
    elif score >= 60:
        return {
            "resolution": "1080p",
            "fps": 30,
            "bitrate": f"{min(conditions.bandwidth * 0.7, 15):.1f} Mbps",
            "codec": "H.264/AVC",
            "streaming_quality": "High"
        }
    elif score >= 40:
        return {
            "resolution": "720p",
            "fps": 30,
            "bitrate": f"{min(conditions.bandwidth * 0.6, 8):.1f} Mbps",
            "codec": "H.264/AVC",
            "streaming_quality": "Medium"
        }
    else:
        return {
            "resolution": "480p",
            "fps": 24,
            "bitrate": f"{min(conditions.bandwidth * 0.5, 4):.1f} Mbps",
            "codec": "H.264/AVC",
            "streaming_quality": "Low"
        }

@app.get("/")
def read_root():
    return {
        "title": "Network Performance Optimizer",
        "version": "2.0",
        "author": "Your Name",
        "description": "Final Year Project - Advanced Network Optimization System",
        "endpoints": [
            "/analyze - Comprehensive network analysis",
            "/optimize - Network optimization recommendations",
            "/predict - Congestion prediction",
            "/docs - API documentation"
        ]
    }

@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_network_conditions(conditions: NetworkConditions):
    try:
        # Perform network analysis
        analysis = analyze_network(conditions)
        
        # Get video recommendations
        video_rec = get_video_recommendations(conditions, analysis.network_score)
        
        # Calculate optimal bandwidth
        current_throughput = conditions.throughput
        optimal_bandwidth = conditions.bandwidth
        if analysis.network_score < 60:
            optimal_bandwidth = conditions.bandwidth * 1.5
        
        return {
            "timestamp": datetime.now().isoformat(),
            "input_conditions": conditions.dict(),
            "analysis_results": {
                "network_score": analysis.network_score,
                "congestion_level": analysis.congestion_level,
                "performance_rating": analysis.performance_rating,
                "recommendations": analysis.recommendations
            },
            "optimization": {
                "current_bandwidth": conditions.bandwidth,
                "recommended_bandwidth": optimal_bandwidth,
                "potential_improvement": f"{min((optimal_bandwidth/conditions.bandwidth - 1) * 100, 100):.1f}%"
            },
            "video_streaming": video_rec,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_network(conditions: NetworkConditions):
    try:
        analysis = analyze_network(conditions)
        
        # Generate bandwidth optimization steps
        min_bw = conditions.bandwidth * 0.5
        max_bw = conditions.bandwidth * 2.0
        steps = np.linspace(min_bw, max_bw, 10)
        
        optimization_steps = []
        for bw in steps:
            test_conditions = conditions.dict()
            test_conditions['bandwidth'] = float(bw)
            test_analysis = analyze_network(NetworkConditions(**test_conditions))
            
            # Get video recommendations for this bandwidth
            video_rec = get_video_recommendations(NetworkConditions(**test_conditions), test_analysis.network_score)
            
            optimization_steps.append({
                "bandwidth": float(bw),
                "predicted_score": float(test_analysis.network_score),
                "congestion_level": test_analysis.congestion_level,
                "video_quality": video_rec
            })
        
        optimal_step = max(optimization_steps, key=lambda x: x['predicted_score'])
        
        # Calculate congestion reduction
        initial_congestion = 100 - analysis.network_score
        optimal_congestion = 100 - optimal_step['predicted_score']
        congestion_reduction = initial_congestion - optimal_congestion
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_conditions": {
                "bandwidth": conditions.bandwidth,
                "throughput": conditions.throughput,
                "packet_loss": conditions.packet_loss,
                "latency": conditions.latency,
                "jitter": conditions.jitter,
                "current_congestion": f"{initial_congestion:.1f}%",
                "current_score": analysis.network_score
            },
            "optimal_configuration": {
                "bandwidth": optimal_step['bandwidth'],
                "predicted_score": optimal_step['predicted_score'],
                "predicted_congestion": optimal_step['congestion_level'],
                "congestion_reduction": f"{congestion_reduction:.1f}%",
                "video_quality": optimal_step['video_quality'],
                "performance_metrics": {
                    "bandwidth_improvement": f"{((optimal_step['bandwidth']/conditions.bandwidth - 1) * 100):.1f}%",
                    "quality_improvement": f"{(optimal_step['predicted_score'] - analysis.network_score):.1f}%",
                    "network_efficiency": f"{(optimal_step['predicted_score']/optimal_step['bandwidth']):.1f} score/Mbps"
                }
            },
            "step_by_step_optimization": optimization_steps,
            "recommendations": analysis.recommendations,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
