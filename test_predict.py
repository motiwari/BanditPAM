#!/usr/bin/env python3
"""
Test script for the predict() function implementation.
Tests Issue #185: Users need a predict() method to assign new data points 
to existing clusters without refitting.
"""

import numpy as np
import sys
import os

# Add the current directory to Python path to import banditpam
sys.path.insert(0, '.')

try:
    import banditpam
    print("✓ Successfully imported banditpam")
except ImportError as e:
    print(f"✗ Failed to import banditpam: {e}")
    print("Make sure to build the package first with: python setup.py build_ext --inplace")
    sys.exit(1)

def test_predict_functionality():
    """Test the predict() function with synthetic data."""
    print("\n=== Testing Predict Functionality ===")
    
    # Create synthetic training data with 3 clear clusters
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    
    # Generate 3 clusters
    cluster1 = np.random.normal([0, 0], 0.5, (n_samples//3, n_features))
    cluster2 = np.random.normal([5, 5], 0.5, (n_samples//3, n_features))
    cluster3 = np.random.normal([10, 0], 0.5, (n_samples//3, n_features))
    
    X_train = np.vstack([cluster1, cluster2, cluster3]).astype(np.float64)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training data type: {X_train.dtype}")
    
    # Create test data points that should be clearly assigned to clusters
    X_test = np.array([
        [0.1, 0.1],    # Should go to cluster 0 (near cluster1)
        [5.1, 5.1],    # Should go to cluster 1 (near cluster2)
        [10.1, 0.1],   # Should go to cluster 2 (near cluster3)
        [2.5, 2.5],    # Should go to cluster 0 (between cluster1 and cluster2, closer to cluster1)
    ]).astype(np.float64)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test data:\n{X_test}")
    
    # Test 1: Basic predict functionality
    print("\n--- Test 1: Basic Predict Functionality ---")
    try:
        # Create and fit the model
        kmedoids = banditpam.KMedoids(n_medoids=3, algorithm="BanditPAM")
        print("✓ Created KMedoids object")
        
        # Fit the model
        kmedoids.fit(X_train, "L2")
        print("✓ Successfully fitted the model")
        
        # Get the final medoids
        medoids = kmedoids.medoids
        print(f"✓ Final medoids: {medoids}")
        
        # Test predict function
        kmedoids.predict(X_test)
        print("✓ Successfully called predict()")
        
        # Get predicted labels
        labels = kmedoids.labels_predict
        print(f"✓ Predicted labels: {labels}")
        
        # Validate predictions make sense
        expected_labels = [0, 1, 2, 0]  # Based on our test data
        print(f"Expected labels: {expected_labels}")
        
        # Check if predictions are reasonable (allowing some flexibility)
        correct_predictions = 0
        for i, (predicted, expected) in enumerate(zip(labels, expected_labels)):
            if predicted == expected:
                correct_predictions += 1
                print(f"  Point {i}: ✓ Correctly assigned to cluster {predicted}")
            else:
                print(f"  Point {i}: ⚠ Assigned to cluster {predicted}, expected {expected}")
        
        accuracy = correct_predictions / len(labels)
        print(f"Prediction accuracy: {accuracy:.2%}")
        
        if accuracy >= 0.75:  # Allow some flexibility due to randomness
            print("✓ Predict function is working correctly!")
            return True
        else:
            print("⚠ Predict function may need adjustment")
            return False
            
    except Exception as e:
        print(f"✗ Error in basic predict test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling in predict function."""
    print("\n--- Test 2: Error Handling ---")
    
    try:
        kmedoids = banditpam.KMedoids(n_medoids=3)
        
        # Test 1: Predict without fitting
        try:
            X_test = np.random.random((5, 2)).astype(np.float64)
            kmedoids.predict(X_test)
            print("✗ Should have raised error for unfitted model")
            return False
        except Exception as e:
            print(f"✓ Correctly raised error for unfitted model: {type(e).__name__}")
        
        # Test 2: Wrong feature dimensions
        try:
            X_train = np.random.random((50, 3)).astype(np.float64)
            X_test = np.random.random((5, 2)).astype(np.float64)  # Wrong number of features
            
            kmedoids.fit(X_train, "L2")
            kmedoids.predict(X_test)
            print("✗ Should have raised error for wrong feature dimensions")
            return False
        except Exception as e:
            print(f"✓ Correctly raised error for wrong dimensions: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in error handling test: {e}")
        return False

def test_different_loss_functions():
    """Test predict with different loss functions."""
    print("\n--- Test 3: Different Loss Functions ---")
    
    # Create simple data
    X_train = np.array([
        [0, 0], [1, 1], [10, 10], [11, 11]
    ]).astype(np.float64)
    
    X_test = np.array([
        [0.5, 0.5], [10.5, 10.5]
    ]).astype(np.float64)
    
    loss_functions = ["L1", "L2", "euclidean", "manhattan"]
    
    for loss_fn in loss_functions:
        try:
            print(f"\nTesting with {loss_fn} loss function:")
            kmedoids = banditpam.KMedoids(n_medoids=2)
            kmedoids.fit(X_train, loss_fn)
            kmedoids.predict(X_test)
            labels = kmedoids.labels_predict
            print(f"  ✓ Predictions: {labels}")
        except Exception as e:
            print(f"  ✗ Error with {loss_fn}: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("BanditPAM Predict Function Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_predict_functionality():
        tests_passed += 1
    
    if test_error_handling():
        tests_passed += 1
        
    if test_different_loss_functions():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The predict() function is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)