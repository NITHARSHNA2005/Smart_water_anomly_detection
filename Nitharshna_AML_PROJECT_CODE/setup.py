#!/usr/bin/env python3
"""
Setup script for Smart Water Analytics
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def check_data_files():
    """Check if data files exist"""
    required_files = [
        "water-consumption-data-main/minutely-water-consumption-data.csv",
        "water_leak_detection_1000_rows.csv"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")

def main():
    print("🌊 Smart Water Analytics Setup")
    print("=" * 40)
    
    # Install requirements
    install_requirements()
    
    # Check data files
    print("\nChecking data files...")
    check_data_files()
    
    

if __name__ == "__main__":
    main()