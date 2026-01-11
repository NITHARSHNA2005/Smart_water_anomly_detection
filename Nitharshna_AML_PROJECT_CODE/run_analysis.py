#!/usr/bin/env python3
"""
Smart Water Analytics - Complete Pipeline Runner
Advanced Deep Learning for Anomaly Detection and Demand Forecasting
"""

from water_analytics import WaterAnalytics
from advanced_models import AdvancedWaterModels
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run the complete smart water analytics pipeline"""
    
    print("🌊" + "="*60 + "🌊")
    print("     SMART WATER ANALYTICS - DEEP LEARNING PIPELINE")
    print("🌊" + "="*60 + "🌊")
    
    # Initialize analytics system
    analytics = WaterAnalytics()
    
    # Run complete analysis
    results = analytics.run_complete_analysis()
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Print detailed results
    print(f"\n📊 DETAILED RESULTS:")
    print(f"   🔍 Total anomalies detected: {sum(results['anomalies'])}")
    print(f"   📈 Forecast horizon: {len(results['forecasts'])} hours")
    print(f"   🎯 Average forecasted consumption: {sum(results['forecasts'])/len(results['forecasts']):.2f}")
    print(f"   📉 Anomaly detection threshold: {analytics.threshold:.4f}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Run 'streamlit run streamlit_dashboard.py' for interactive dashboard")
    print(f"   2. Check generated visualizations and reports")
    print(f"   3. Monitor real-time predictions and alerts")
    
    return results

if __name__ == "__main__":
    results = main()