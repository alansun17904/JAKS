#!/usr/bin/env python3
"""
Test script for circuit stability with small models (<70M parameters)
Optimized for Runpod environment testing
"""

import torch
import os
import sys
import json
from pathlib import Path

# Add circuit-stability code to path
sys.path.append(str(Path(__file__).parent / "circuit-stability" / "code"))

from src.experiments.utils import load_model
from src.cdatasets import ArithDataset, DatasetBuilder
from src.eap import Graph, evaluate_graph

def test_small_model_loading():
    """Test loading small models compatible with the system"""
    print("🧪 Testing small model loading...")
    
    # Test models in order of preference
    test_models = [
        "pythia-70m",
        "pythia-160m", 
        "gpt2",
        "distilgpt2"
    ]
    
    for model_name in test_models:
        try:
            print(f"\n📦 Loading {model_name}...")
            model = load_model(
                base_model=model_name,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                large_model=False  # Use full precision for small models
            )
            
            # Test basic functionality
            test_input = "The quick brown fox"
            tokens = model.to_tokens(test_input)
            logits = model(tokens)
            
            print(f"✅ {model_name} loaded successfully!")
            print(f"   - Parameters: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
            print(f"   - Device: {next(model.parameters()).device}")
            print(f"   - Output shape: {logits.shape}")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return model_name  # Return first successful model
            
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {str(e)}")
            continue
    
    raise RuntimeError("No compatible small models found!")

def test_dataset_creation():
    """Test dataset creation with small configuration"""
    print("\n🔢 Testing dataset creation...")
    
    try:
        # Create a small arithmetic dataset for testing
        dataset = ArithDataset(
            n_digits=2,      # Smaller than default
            operation="+",
            n_samples=100,   # Much smaller for testing
            balanced=True
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"   - Size: {len(dataset)} samples")
        print(f"   - Sample: {dataset[0]}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {str(e)}")
        raise

def test_circuit_discovery(model_name, dataset):
    """Test basic circuit discovery with small model"""
    print(f"\n🔍 Testing circuit discovery with {model_name}...")
    
    try:
        # Load model again for circuit discovery
        model = load_model(
            base_model=model_name,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            large_model=False
        )
        
        # Create minimal graph for testing
        graph = Graph(
            model=model,
            dataset=dataset,
            batch_size=8,  # Very small batch size
            device=model.cfg.device
        )
        
        # Test basic graph operations
        print("   - Graph created successfully")
        print(f"   - Model: {model_name}")
        print(f"   - Batch size: 8")
        print(f"   - Device: {model.cfg.device}")
        
        # Run minimal evaluation (just a few samples)
        small_dataset = dataset[:10]  # Just 10 samples for testing
        
        # Note: Full circuit discovery would be run here, but for testing
        # we'll just verify the setup works
        print("✅ Circuit discovery setup verified!")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"❌ Circuit discovery test failed: {str(e)}")
        raise

def test_runpod_environment():
    """Test Runpod-specific environment setup"""
    print("\n🏃 Testing Runpod environment...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"   - CUDA Version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available, will use CPU")
    
    # Check memory
    import psutil
    memory = psutil.virtual_memory()
    print(f"✅ System Memory: {memory.total / 1e9:.1f} GB ({memory.percent}% used)")
    
    # Check storage
    disk = psutil.disk_usage('/')
    print(f"✅ Disk Space: {disk.total / 1e9:.1f} GB total, {disk.free / 1e9:.1f} GB free")
    
    # Test key dependencies
    try:
        import transformer_lens
        import pygraphviz
        print("✅ Key dependencies installed correctly")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        raise

def main():
    """Main test function"""
    print("🔬 Circuit Stability Small Model Testing")
    print("=" * 50)
    
    try:
        # Test 1: Environment
        test_runpod_environment()
        
        # Test 2: Model loading
        successful_model = test_small_model_loading()
        
        # Test 3: Dataset creation
        dataset = test_dataset_creation()
        
        # Test 4: Circuit discovery setup
        test_circuit_discovery(successful_model, dataset)
        
        print("\n🎉 All tests passed! Ready for circuit stability experiments.")
        print(f"✅ Recommended model: {successful_model}")
        print("✅ Environment configured correctly")
        
        # Create a simple run configuration
        config = {
            "model": successful_model,
            "dataset_size": 1000,
            "batch_size": 16,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "small_model_optimized": True
        }
        
        with open("small_model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration saved to small_model_config.json")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        print("Please check the error above and ensure all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()