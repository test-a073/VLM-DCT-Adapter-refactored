#!/usr/bin/env python3
"""
Debug script for preference loss computation issues.
This script helps identify where the data mismatch is occurring.
"""

import pickle
import sys
import os

def check_original_results():
    """Check if original_results.pkl exists and inspect its contents."""
    if not os.path.exists('original_results.pkl'):
        print("❌ original_results.pkl not found!")
        print("You need to generate it first by:")
        print("1. Setting CREATE_ORIGINAL_RESPONSE_PKL_FILE=True in main_llama2_injection.py")
        print("2. Running the training script to generate original results")
        return False
    
    try:
        with open('original_results.pkl', 'rb') as f:
            original_results = pickle.load(f)
        
        print(f"✅ Found original_results.pkl with {len(original_results)} entries")
        
        # Show first few entries
        print("\nFirst 3 original results (truncated):")
        for i, result in enumerate(original_results[:3]):
            print(f"  [{i}]: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading original_results.pkl: {e}")
        return False

def check_dataset_alignment():
    """Check if the dataset and original results are aligned."""
    print("\n" + "="*50)
    print("DATASET ALIGNMENT CHECK")
    print("="*50)
    
    # This would need to be implemented based on your specific dataset loading
    print("TODO: Implement dataset alignment check")
    print("You need to verify that:")
    print("- The dataset order matches the order used when generating original_results.pkl")
    print("- The batch processing follows the same pattern")
    print("- shuffle=False is used consistently")

def main():
    print("🔍 DEBUGGING PREFERENCE LOSS COMPUTATION")
    print("="*60)
    
    # Check original results
    if not check_original_results():
        return
    
    # Check dataset alignment
    check_dataset_alignment()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. If original_results.pkl is missing, generate it first")
    print("2. Run training with DEBUG=True in compute_preference_loss")
    print("3. Check the debug output for data alignment issues")
    print("4. Verify model generation is working correctly")

if __name__ == "__main__":
    main() 