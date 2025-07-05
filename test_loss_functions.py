#!/usr/bin/env python3
"""
Test script to verify loss function parsing works correctly.
This tests the C++ setLossFn function through the Python bindings.
"""

import numpy as np

def test_loss_functions():
    """Test various loss functions to ensure they work correctly."""
    
    # Generate some test data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    
    print("Testing loss function parsing...")
    
    # Test cases for different loss functions
    test_cases = [
        ("L2", "L2 norm"),
        ("l2", "lowercase L2 norm"),
        ("L1", "L1 norm"),
        ("l1", "lowercase L1 norm"),
        ("L3", "L3 norm"),
        ("manhattan", "Manhattan distance"),
        ("cosine", "Cosine distance"),
        ("cos", "Cosine distance (short)"),
        ("inf", "L-infinity norm"),
        ("euclidean", "Euclidean distance"),
    ]
    
    try:
        from banditpam import KMedoids
        
        for loss_func, description in test_cases:
            print(f"Testing {loss_func} ({description})...")
            
            # Create KMedoids object
            kmed = KMedoids(n_medoids=3, algorithm="BanditPAM")
            
            # Set the loss function
            kmed.loss_function = loss_func
            
            # Try to fit the data
            kmed.fit(X, loss_func)
            
            print(f"  ✅ {loss_func} works correctly")
            
        print("\n🎉 All loss functions tested successfully!")
        
    except ImportError:
        print("❌ Could not import banditpam. Build the package first.")
        print("Try: python3 -m pip install -e .")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

def test_invalid_loss_function():
    """Test that invalid loss functions are properly rejected."""
    
    print("\nTesting invalid loss functions...")
    
    try:
        from banditpam import KMedoids
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        
        # Test invalid loss functions
        invalid_losses = ["invalid", "l0", "l-1", "unknown"]
        
        for invalid_loss in invalid_losses:
            print(f"Testing invalid loss: {invalid_loss}...")
            
            try:
                kmed = KMedoids(n_medoids=3, algorithm="BanditPAM")
                kmed.fit(X, invalid_loss)
                print(f"  ❌ {invalid_loss} should have failed but didn't")
            except Exception as e:
                print(f"  ✅ {invalid_loss} correctly rejected: {str(e)[:50]}...")
                
    except ImportError:
        print("❌ Could not import banditpam. Build the package first.")

if __name__ == "__main__":
    print("🧪 Testing BanditPAM Loss Function Parsing")
    print("=" * 50)
    
    test_loss_functions()
    test_invalid_loss_function()
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!") 