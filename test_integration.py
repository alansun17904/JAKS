#!/usr/bin/env python3
"""
Test script for ToT-Circuit Stability integration
"""

import subprocess
import json
from pathlib import Path
import sys

def simulate_tot_call():
    """
    Simulate calling the circuit stability pipeline from ToT
    """
    print("🧪 Testing ToT-Circuit Stability Integration")
    
    # Step 1: Create a sample ToT data file (simulating what ToT would generate)
    tot_data = {
        "data_entry": "1 1 4 6",
        "steps": [
            {
                "step": 0,
                "Prompt": "Input: 2 8 8 14\nPossible next steps:\n2 + 8 = 10 (left: 8 10 14)\nInput: 1 1 4 6\nPossible next steps:\n",
                "raw_output_prop": [],
                "raw_output_eval": [],
                "thought_variation": {
                    "1 + 1 = 2 (left: 2 4 6)": [0.9],
                    "1 * 4 = 4 (left: 1 4 6)": [0.7],
                    "4 + 6 = 10 (left: 1 1 10)": [0.8],
                    "6 - 4 = 2 (left: 1 1 2)": [0.6],
                    "1 + 6 = 7 (left: 1 4 7)": [0.5]
                }
            }
        ]
    }
    
    # Step 2: Save ToT data to the expected location
    data_dir = Path("circuit-stability/code/src/cdatasets/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    tot_file = data_dir / "1,1,4,6.json"
    with open(tot_file, 'w') as f:
        json.dump(tot_data, f, indent=2)
    
    print(f"✓ Created ToT data file: {tot_file}")
    
    # Step 3: Run circuit stability analysis (simulating what happens in bfs.py)
    print("🔬 Running circuit stability analysis...")
    
    try:
        # Need to run from the circuit-stability/code directory with proper environment
        # Default to GPU for better performance
        result = subprocess.run([
            "bash", "-c",
            "source ../activate_environment.sh && bash src/scripts/naive_run.sh --data_params data_file=1,1,4,6.json"
        ], 
        check=True, 
        capture_output=True, 
        text=True,
        cwd="/workspace/JAKS/circuit-stability/code",
        shell=False
        )
        
        print("✓ Circuit analysis completed successfully")
        print(f"Output: {result.stdout[-500:]}")  # Show last 500 chars
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Circuit analysis failed: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    
    # Step 4: Test get_circuit_scores function
    print("📊 Testing circuit score parsing...")
    
    # Import our test function
    sys.path.append("/workspace/JAKS")
    from test_circuit_scores import get_circuit_scores
    
    # Test with the thought variations from our ToT data
    thought_variations = list(tot_data["steps"][0]["thought_variation"].keys())
    scores = get_circuit_scores(None, "1 1 4 6", thought_variations)
    
    print(f"✓ Generated scores for {len(thought_variations)} variations:")
    for i, (variation, score) in enumerate(zip(thought_variations, scores)):
        print(f"   {i+1}. {variation}: {score:.4f}")
    
    # Step 5: Simulate ToT selection logic
    print("\n🎯 Simulating ToT thought selection...")
    
    # Sort by score (highest first)
    scored_variations = list(zip(thought_variations, scores))
    scored_variations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top-ranked thoughts (by circuit stability):")
    for i, (variation, score) in enumerate(scored_variations[:3]):
        print(f"   {i+1}. {variation} (score: {score:.4f})")
    
    print("\n🎉 Integration test completed successfully!")
    return True

if __name__ == "__main__":
    success = simulate_tot_call()
    sys.exit(0 if success else 1)