#!/usr/bin/env python3
"""
Test script for get_circuit_scores function
"""

import json
from pathlib import Path

def get_circuit_scores(task, x, ys):
    """
    Test version of get_circuit_scores function
    """
    scores = []
    n_thoughts = 5  # Should match the value in circuit_discovery.py
    
    # The circuit analysis generates score files in the circuit-stability/code directory
    score_dir = Path("circuit-stability/code/")
    
    # Find all generated score files
    score_files = list(score_dir.glob("*_score.json"))
    score_files.sort()  # Ensure consistent ordering
    
    print(f"Found {len(score_files)} circuit score files")
    print(f"Need scores for {len(ys)} thought variations")
    
    # If we have score files, read them
    if score_files:
        for i, y in enumerate(ys):
            try:
                # Calculate which score file corresponds to this variation
                file_idx = i % len(score_files) if score_files else 0
                score_file = score_files[file_idx]
                
                with open(score_file, 'r') as f:
                    score_data = json.load(f)
                
                # Use circuit loss as the score (lower is better, so we negate it)
                # Convert to the same scale as heuristic scores (0.001 to 20)
                circuit_loss = score_data.get("circuit_loss", 0)
                
                # Transform circuit loss to heuristic score range
                # Good circuits have lower loss, so we invert and scale
                # Auto-scale based on the range of losses we're seeing
                
                if circuit_loss <= 0:
                    score = 20  # Perfect score for no extra loss
                else:
                    # For positive losses, use relative ranking
                    # Lower loss = higher score
                    # Map the range dynamically
                    min_loss = 25000  # Approximate minimum we're seeing
                    max_loss = 50000  # Approximate maximum we're seeing
                    
                    if circuit_loss <= min_loss:
                        score = 20
                    elif circuit_loss >= max_loss:
                        score = 0.001
                    else:
                        # Linear mapping from [min_loss, max_loss] to [20, 0.001]
                        normalized = (circuit_loss - min_loss) / (max_loss - min_loss)
                        score = 20 - normalized * 19.999
                
                scores.append(score)
                print(f"Variation {i}: circuit_loss={circuit_loss:.4f} -> score={score:.4f}")
                
            except Exception as e:
                print(f"Error reading score file {score_file}: {e}")
                scores.append(1.0)  # Default middle score
    else:
        # Fallback: if no score files found, use default scores
        print("No circuit score files found, using default scores")
        scores = [1.0] * len(ys)
    
    return scores

if __name__ == "__main__":
    # Test the function
    x = "1 1 4 6"
    ys = ["1 + 1 = 2", "1 * 4 = 4", "6 - 4 = 2"]
    
    scores = get_circuit_scores(None, x, ys)
    print(f"\nFinal scores: {scores}")