#!/usr/bin/env python
"""
Master script to run all assignment models
Run from: /Users/vivekgowdas/Desktop/LLM/assignment/
"""

import subprocess
import sys
import os

def run_model(name, path):
    """Run a model script"""
    print("\n" + "="*80)
    print(f"Running: {name}")
    print("="*80)
    
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, path], check=True)
        print(f"✅ {name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {name}: {e}")
        return False

def main():
    print("="*80)
    print("MACHINE LEARNING ASSIGNMENT - RUNNING ALL MODELS")
    print("="*80)
    
    results = []
    
    # Run Binary Classification
    results.append(run_model(
        "Binary Classification ANN",
        "ANN/Binary/binary_classification.py"
    ))
    
    # Run Multiclass Classification
    results.append(run_model(
        "Multiclass Classification ANN",
        "ANN/Multiclass/multiclass_classification.py"
    ))
    
    # Run Logistic Regression
    results.append(run_model(
        "Logistic Regression",
        "LR/logistic_regression.py"
    ))
    
    # Summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    models = [
        "✅ Binary Classification ANN" if results[0] else "❌ Binary Classification ANN",
        "✅ Multiclass Classification ANN" if results[1] else "❌ Multiclass Classification ANN",
        "✅ Logistic Regression" if results[2] else "❌ Logistic Regression"
    ]
    
    for model in models:
        print(model)
    
    print("\n" + "="*80)
    if all(results):
        print("🎉 All models executed successfully!")
    else:
        print("⚠️  Some models failed. Check output above.")
    print("="*80)

if __name__ == "__main__":
    main()
