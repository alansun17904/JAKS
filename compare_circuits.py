import json
import os
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict

def get_edge_scores_from_file(file_path):
    """Loads a circuit JSON file and returns a list of its edge scores."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract the 'score' from each edge in the 'edges' list
    scores = []
    edge_scores = {}
    for edge_key, edge_data in data['edges'].items():
        scores.append(edge_data['score'])
        edge_scores[edge_key] = edge_data['score']
    return scores, edge_scores, data

def get_hard_circuit_from_file(file_path):
    """Loads a circuit JSON file and returns the set of edges in the hard circuit."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
        
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract edges that are part of the hard circuit (in_graph=True)
    hard_circuit = {edge_key for edge_key, edge_data in data['edges'].items() if edge_data.get('in_graph', False)}
    return hard_circuit

def visualize_score_comparison(scores1, scores2, output_dir='.', show=False):
    """
    Create a comparative visualization of edge scores from two circuits.
    
    Benefits:
    - Shows all scores from both circuits as separate data series
    - Makes it easy to identify edges with high/low scores in each circuit
    - Highlights differences between circuits across all edges
    
    Drawbacks:
    - Doesn't directly show correlation between scores
    - Edge indices don't correspond to specific edge identities
    """
    plt.figure(figsize=(10, 8))
    
    # Create index for x-axis
    x = list(range(len(scores1)))
    
    # Sort the scores to make visualization clearer
    sorted_indices = sorted(range(len(scores1)), key=lambda i: scores1[i])
    sorted_scores1 = [scores1[i] for i in sorted_indices]
    sorted_scores2 = [scores2[i] for i in sorted_indices]
    
    # Plot Circuit 1 scores in blue
    plt.scatter(x, sorted_scores1, alpha=0.5, color='blue', label='Circuit 1')
    # Plot Circuit 2 scores in red
    plt.scatter(x, sorted_scores2, alpha=0.5, color='red', label='Circuit 2')
    
    plt.xlabel('Edge Index')
    plt.ylabel('Edge Scores')
    plt.title('Edge Score Comparison between Circuits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'score_comparison.png')
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    
    print(f"Score comparison plot saved to {output_path}")

def visualize_score_correlation(scores1, scores2, output_dir='.', show=False):
    """
    Create a scatter plot showing correlation between edge scores from two circuits.
    
    Benefits:
    - Directly visualizes correlation between edge importance in two circuits
    - Helps identify outliers where edge importance drastically differs
    
    Drawbacks:
    - Can be cluttered with many edges
    - Doesn't show edge identity (which specific edge is which point)
    """
    plt.figure(figsize=(10, 8))
    
    # Create two scatter plots:
    # 1. Red dots for Circuit 1 edges (x-axis is Circuit 1 scores, y-axis is 0)
    # 2. Blue dots for Circuit 2 edges (x-axis is 0, y-axis is Circuit 2 scores)
    plt.scatter(scores1, np.zeros_like(scores1), alpha=0.6, color='red', label='Circuit 1 Edges')
    plt.scatter(np.zeros_like(scores2), scores2, alpha=0.6, color='blue', label='Circuit 2 Edges')
    
    # Add correlation line in green
    if len(scores1) == len(scores2):
        m, b = np.polyfit(scores1, scores2, 1)
        plt.plot(scores1, m*np.array(scores1) + b, color='green', linestyle='--', label='Correlation')
    
    plt.xlabel('Circuit 1 Edge Scores')
    plt.ylabel('Circuit 2 Edge Scores')
    plt.title('Edge Score Correlation between Circuits')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'score_correlation.png')
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    
    print(f"Score correlation plot saved to {output_path}")

def visualize_score_distributions(scores1, scores2, output_dir='.', show=False):
    """
    Create histograms of edge scores for both circuits.
    
    Benefits:
    - Shows the distribution of edge importance in each circuit
    - Can reveal if one circuit has more high/low importance edges
    
    Drawbacks:
    - Doesn't show relationship between specific edges across circuits
    - Only gives aggregate information
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(scores1, bins=20, alpha=0.7, color='blue')
    plt.title('Circuit 1 Edge Score Distribution')
    plt.xlabel('Edge Score')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(scores2, bins=20, alpha=0.7, color='green')
    plt.title('Circuit 2 Edge Score Distribution')
    plt.xlabel('Edge Score')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    
    print(f"Score distributions plot saved to {output_path}")

def visualize_edge_rank_changes(edge_scores1, edge_scores2, output_dir='.', top_n=30, show=False):
    """
    Create a visualization of rank changes for the top N edges.
    
    Benefits:
    - Clearly shows how edge importance rankings change between circuits
    - Focuses on most important edges that drive model behavior
    
    Drawbacks:
    - Limited to a subset of edges (top N)
    - May not show interesting patterns in lower-ranked edges
    """
    # Convert scores to rankings
    edges1 = sorted(edge_scores1.items(), key=lambda x: x[1], reverse=True)
    edges2 = sorted(edge_scores2.items(), key=lambda x: x[1], reverse=True)
    
    # Get top N edges from both rankings
    top_edges = set([e[0] for e in edges1[:top_n]] + [e[0] for e in edges2[:top_n]])
    
    # Create rank dictionaries
    ranks1 = {edge: rank for rank, (edge, _) in enumerate(edges1, 1)}
    ranks2 = {edge: rank for rank, (edge, _) in enumerate(edges2, 1)}
    
    # Get ranks for top edges
    top_edges_ranks = []
    for edge in top_edges:
        rank1 = ranks1.get(edge, len(edges1) + 1)  # Default to last+1 if not found
        rank2 = ranks2.get(edge, len(edges2) + 1)
        edge_label = edge[:20] + '...' if len(edge) > 20 else edge
        top_edges_ranks.append((edge_label, rank1, rank2))
    
    # Sort by average rank
    top_edges_ranks.sort(key=lambda x: (x[1] + x[2]) / 2)
    
    # Plot
    plt.figure(figsize=(12, max(8, len(top_edges_ranks) * 0.4)))
    
    labels = [item[0] for item in top_edges_ranks]
    rank1 = [item[1] for item in top_edges_ranks]
    rank2 = [item[2] for item in top_edges_ranks]
    
    y_pos = range(len(labels))
    
    plt.hlines(y=y_pos, xmin=rank1, xmax=rank2, color='grey', alpha=0.5)
    plt.scatter(rank1, y_pos, color='blue', alpha=0.7, label='Circuit 1 Rank')
    plt.scatter(rank2, y_pos, color='green', alpha=0.7, label='Circuit 2 Rank')
    
    plt.yticks(y_pos, labels)
    plt.xlabel('Rank (lower is more important)')
    plt.title(f'Edge Rank Changes (Top {len(top_edges_ranks)} Edges)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rank_changes.png')
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    
    print(f"Rank changes plot saved to {output_path}")

def visualize_hard_circuit_comparison(hard_circuit1, hard_circuit2, edge_scores1, edge_scores2, output_dir='.', show=False):
    """
    Create a Venn diagram-like visualization showing overlap between hard circuits.
    
    Benefits:
    - Directly visualizes the intersection and differences between hard circuits
    - Shows the size of each circuit and their overlap
    
    Drawbacks:
    - Doesn't show the specific identity of each edge
    - No structural information about the circuits
    """
    plt.figure(figsize=(10, 8))
    
    # Get sets of edges
    set1 = set(hard_circuit1)
    set2 = set(hard_circuit2)
    
    # Create 3 sets: unique to set1, intersection, unique to set2
    only_in_1 = set1 - set2
    intersection = set1 & set2
    only_in_2 = set2 - set1
    
    # Calculate average scores for each region
    avg_score_only1 = sum(edge_scores1.get(edge, 0) for edge in only_in_1) / (len(only_in_1) if only_in_1 else 1)
    avg_score_inter = sum((edge_scores1.get(edge, 0) + edge_scores2.get(edge, 0))/2 for edge in intersection) / (len(intersection) if intersection else 1)
    avg_score_only2 = sum(edge_scores2.get(edge, 0) for edge in only_in_2) / (len(only_in_2) if only_in_2 else 1)
    
    # Custom Venn diagram
    plt.subplot(1, 1, 1)
    venn = plt.axes([0.1, 0.1, 0.8, 0.8])
    venn.set_xlim(-1.5, 1.5)
    venn.set_ylim(-1.5, 1.5)
    
    # Plot circles
    c1 = plt.Circle((-0.5, 0), 1.0, alpha=0.5, color='blue')
    c2 = plt.Circle((0.5, 0), 1.0, alpha=0.5, color='green')
    venn.add_patch(c1)
    venn.add_patch(c2)
    
    # Add text
    plt.text(-0.8, 0, f"Only Circuit 1\n{len(only_in_1)} edges\nAvg score: {avg_score_only1:.3f}", 
             ha='center', va='center', fontsize=11)
    plt.text(0.0, 0, f"Intersection\n{len(intersection)} edges\nAvg score: {avg_score_inter:.3f}", 
             ha='center', va='center', fontsize=11)
    plt.text(0.8, 0, f"Only Circuit 2\n{len(only_in_2)} edges\nAvg score: {avg_score_only2:.3f}", 
             ha='center', va='center', fontsize=11)
    
    plt.title('Hard Circuit Comparison')
    plt.axis('off')
    
    output_path = os.path.join(output_dir, 'hard_circuit_comparison.png')
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    
    print(f"Hard circuit comparison plot saved to {output_path}")

def visualize_circuit_graphs(data1, data2, edge_scores1, edge_scores2, hard_circuit1, hard_circuit2, output_dir='.', show=False):
    """
    Create a network visualization of both circuits with highlighted similarities and differences.
    
    Benefits:
    - Shows the actual structure of the circuits
    - Highlights common and unique components visually
    - Reveals structural patterns that may not be apparent in statistics
    
    Drawbacks:
    - Can be complex and hard to interpret for large circuits
    - May require manual layout adjustments for clarity
    - Computationally expensive for very large circuits
    """
    # Create a combined graph
    G = nx.DiGraph()
    
    # Function to add edges to the graph
    def add_edges_from_data(data, hard_circuit, is_circuit1):
        circuit_id = 1 if is_circuit1 else 2
        prefix = f"C{circuit_id}_"
        
        # Process nodes and edges from the data
        if 'nodes' in data and isinstance(data['nodes'], dict):
            for node_key, node_data in data['nodes'].items():
                # Add node with attributes - handle case where node_data might be a boolean or other type
                label = node_key
                if isinstance(node_data, dict) and 'label' in node_data:
                    label = node_data['label']
                
                G.add_node(node_key, 
                          label=label,
                          in_circuit1=is_circuit1,
                          in_circuit2=not is_circuit1)
        
        if 'edges' in data and isinstance(data['edges'], dict):
            for edge_key, edge_data in data['edges'].items():
                # Check if this edge is in the hard circuit
                in_hard = edge_key in hard_circuit
                
                # Extract source and target from edge_key
                # Assuming edge_key format contains source and target
                parts = edge_key.split('->')
                if len(parts) != 2:
                    continue
                    
                source, target = parts[0].strip(), parts[1].strip()
                
                # Add edge with attributes
                G.add_edge(source, target, 
                          key=edge_key,
                          in_circuit1=is_circuit1,
                          in_circuit2=not is_circuit1,
                          in_hard1=in_hard if is_circuit1 else False,
                          in_hard2=in_hard if not is_circuit1 else False,
                          score1=edge_scores1.get(edge_key, 0) if is_circuit1 else 0,
                          score2=edge_scores2.get(edge_key, 0) if not is_circuit1 else 0)
    
    try:
        # Add edges from both circuits
        add_edges_from_data(data1, hard_circuit1, True)
        add_edges_from_data(data2, hard_circuit2, False)
        
        # Skip visualization if the graph is empty
        if len(G.nodes()) == 0:
            print("Warning: No nodes found in the graph. Skipping circuit graph visualization.")
            return
            
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Define edge colors based on which circuit they belong to
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            if data.get('in_hard1', False) and data.get('in_hard2', False):
                # In both hard circuits - purple
                edge_colors.append('purple')
                edge_widths.append(2.5)
            elif data.get('in_hard1', False):
                # Only in circuit 1 hard circuit - blue
                edge_colors.append('blue')
                edge_widths.append(1.5)
            elif data.get('in_hard2', False):
                # Only in circuit 2 hard circuit - green
                edge_colors.append('green')
                edge_widths.append(1.5)
            else:
                # In neither hard circuit - light gray
                edge_colors.append('lightgray')
                edge_widths.append(0.5)
        
        # Define node colors
        node_colors = []
        for node in G.nodes():
            if G.nodes[node].get('in_circuit1', False) and G.nodes[node].get('in_circuit2', False):
                # In both circuits - orange
                node_colors.append('orange')
            elif G.nodes[node].get('in_circuit1', False):
                # Only in circuit 1 - light blue
                node_colors.append('lightblue')
            else:
                # Only in circuit 2 - light green
                node_colors.append('lightgreen')
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)  # Consistent layout
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6, 
                            arrowsize=10, connectionstyle='arc3,rad=0.1')
        
        # Add minimal labels for clarity (may be cluttered with large graphs)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title('Combined Circuit Graph Visualization')
        plt.axis('off')
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='purple', lw=2.5, label='In Both Hard Circuits'),
            Line2D([0], [0], color='blue', lw=1.5, label='Only in Circuit 1 Hard'),
            Line2D([0], [0], color='green', lw=1.5, label='Only in Circuit 2 Hard'),
            Line2D([0], [0], color='lightgray', lw=0.5, label='In Neither Hard Circuit'),
            Line2D([0], [0], marker='o', color='orange', label='Node in Both Circuits', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='o', color='lightblue', label='Node only in Circuit 1', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='o', color='lightgreen', label='Node only in Circuit 2', markersize=10, linestyle='None')
        ]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'combined_circuit_graph.png')
        plt.savefig(output_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        print(f"Combined circuit graph visualization saved to {output_path}")
    except Exception as e:
        print(f"Error creating circuit graph visualization: {e}")
        print("Skipping circuit graph visualization.")

def visualize_heatmap_comparison(edge_scores1, edge_scores2, hard_circuit1, hard_circuit2, output_dir='.', show=False):
    """
    Create a heatmap visualization comparing edge scores between circuits.
    
    Benefits:
    - Clear visual encoding of score differences
    - Highlights patterns in how scores differ between circuits
    - Can reveal clusters of similarly behaving edges
    
    Drawbacks:
    - Can be hard to read with many edges
    - May require filtering to focus on important edges
    """
    # Get union of all edges from both circuits
    all_edges = set(edge_scores1.keys()).union(set(edge_scores2.keys()))
    
    # Create data for the heatmap
    edge_labels = []
    scores_matrix = []
    
    # Get top edges by maximum score in either circuit
    edge_max_scores = {edge: max(edge_scores1.get(edge, 0), edge_scores2.get(edge, 0)) for edge in all_edges}
    top_edges = sorted(edge_max_scores.items(), key=lambda x: x[1], reverse=True)[:50]  # Limit to top 50 edges
    
    for edge, _ in top_edges:
        edge_labels.append(edge[:20] + '...' if len(edge) > 20 else edge)
        
        # Get scores and hard circuit membership
        score1 = edge_scores1.get(edge, 0)
        score2 = edge_scores2.get(edge, 0)
        in_hard1 = edge in hard_circuit1
        in_hard2 = edge in hard_circuit2
        
        # Create row: [Score1, Score2, In Hard1, In Hard2]
        scores_matrix.append([score1, score2, 1 if in_hard1 else 0, 1 if in_hard2 else 0])
    
    # Convert to numpy array
    scores_array = np.array(scores_matrix)
    
    # Create heatmap
    plt.figure(figsize=(12, max(8, len(top_edges) * 0.4)))
    
    # Create labels for columns
    col_labels = ['Score C1', 'Score C2', 'Hard C1', 'Hard C2']
    
    # Create custom colormap that transitions from white to blue
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', 'blue'], N=100)
    
    # Plot heatmap
    ax = sns.heatmap(scores_array, cmap=cmap, annot=True, fmt='.2f',
                   xticklabels=col_labels, yticklabels=edge_labels)
    
    # Adjust y-tick label size based on number of edges
    plt.yticks(rotation=0, fontsize=max(4, 10 - len(top_edges) // 10))
    
    plt.title('Edge Score Comparison Heatmap (Top Edges)')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'score_heatmap.png')
    plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()
    
    print(f"Score heatmap visualization saved to {output_path}")

def main():
    """
    Main function that compares two circuits and generates selected visualizations.
    
    Args:
        visualizations: List of visualizations to generate. If empty, all visualizations are generated.
                      Options: 'correlation', 'distributions', 'rank_changes', 'hard_comparison',
                              'circuit_graphs', 'heatmap'
        output_dir: Directory to save visualization outputs
        show_plots: Whether to display the plots in addition to saving them
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Compare neural circuits and generate visualizations.')
    parser.add_argument('--visualizations', nargs='*', default=[], 
                      help='List of visualizations to generate. If empty, all are generated.')
    parser.add_argument('--output_dir', type=str, default='.',
                      help='Directory to save visualization outputs.')
    parser.add_argument('--show', action='store_true',
                      help='Display plots in addition to saving them.')
    
    args = parser.parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths - use absolute paths
    file1_path = os.path.join(script_dir, 'output.pkl.json')
    file2_path = os.path.join(script_dir, 'output_paren.json')

    print(f"Comparing circuits from:\n1: {file1_path}\n2: {file2_path}\n")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # SOFT CIRCUIT COMPARISON (Spearman's Rank Correlation)
    scores1, edge_scores1, data1 = get_edge_scores_from_file(file1_path)
    scores2, edge_scores2, data2 = get_edge_scores_from_file(file2_path)
    
    if scores1 is None or scores2 is None:
        print("Could not proceed with soft circuit analysis due to missing file(s).")
        return
    
    if len(scores1) != len(scores2):
        print(f"Warning: Circuits have different number of edges ({len(scores1)} vs {len(scores2)}).")
        print("This may affect correlation calculations.")
    
    correlation, p_value = spearmanr(scores1, scores2)
    print("\n--- Soft Circuit Stability Analysis ---")
    print(f"Spearman's Rank Correlation (ρ): {correlation:.4f}")
    print(f"P-value: {p_value:.4g}")
    
    # Interpretation
    print("\nInterpretation:")
    if correlation > 0.8:
        print("Result: Very High Stability. The model uses very similar reasoning processes.")
    elif correlation > 0.6:
        print("Result: High Stability. The model uses similar reasoning processes.")
    elif correlation > 0.4:
        print("Result: Moderate Stability. There are notable similarities in reasoning.")
    elif correlation > 0.2:
        print("Result: Low Stability. The reasoning processes are largely different.")
    else:
        print("Result: Very Low or No Stability. The model uses fundamentally different reasoning processes.")
    
    # HARD CIRCUIT COMPARISON (Jaccard Similarity)
    hard_circuit1 = get_hard_circuit_from_file(file1_path)
    hard_circuit2 = get_hard_circuit_from_file(file2_path)
    
    if hard_circuit1 is None or hard_circuit2 is None:
        print("\nCould not proceed with hard circuit analysis due to missing file(s).")
        return
    
    # Calculate Jaccard similarity
    intersection = len(hard_circuit1.intersection(hard_circuit2))
    union = len(hard_circuit1.union(hard_circuit2))
    jaccard = intersection / union if union > 0 else 0
    
    print("\n--- Hard Circuit Stability Analysis ---")
    print(f"Hard Circuit 1 Size: {len(hard_circuit1)} edges")
    print(f"Hard Circuit 2 Size: {len(hard_circuit2)} edges")
    print(f"Intersection: {intersection} edges")
    print(f"Union: {union} edges")
    print(f"Jaccard Similarity: {jaccard:.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if jaccard > 0.8:
        print("Result: Very High Hard Circuit Overlap")
    elif jaccard > 0.6:
        print("Result: High Hard Circuit Overlap")
    elif jaccard > 0.4:
        print("Result: Moderate Hard Circuit Overlap")
    elif jaccard > 0.2:
        print("Result: Low Hard Circuit Overlap")
    else:
        print("Result: Very Low or No Hard Circuit Overlap")
    
    # Define visualization options
    all_visualizations = {
        'correlation': visualize_score_correlation,
        'comparison': visualize_score_comparison,
        'distributions': visualize_score_distributions,
        'rank_changes': visualize_edge_rank_changes,
        'hard_comparison': visualize_hard_circuit_comparison,
        'circuit_graphs': visualize_circuit_graphs,
        'heatmap': visualize_heatmap_comparison
    }
    
    # Determine which visualizations to generate
    visualizations_to_run = args.visualizations
    if not visualizations_to_run:
        visualizations_to_run = list(all_visualizations.keys())
    
    print("\n--- Generating Visualizations ---")
    for viz_name in visualizations_to_run:
        if viz_name not in all_visualizations:
            print(f"Warning: Unknown visualization '{viz_name}', skipping.")
            continue
            
        print(f"Generating {viz_name} visualization...")
        viz_func = all_visualizations[viz_name]
        
        # Call the appropriate visualization function with correct arguments
        if viz_name == 'correlation':
            visualize_score_correlation(scores1, scores2, args.output_dir, args.show)
        elif viz_name == 'comparison':
            visualize_score_comparison(scores1, scores2, args.output_dir, args.show)
        elif viz_name == 'distributions':
            visualize_score_distributions(scores1, scores2, args.output_dir, args.show)
        elif viz_name == 'rank_changes':
            visualize_edge_rank_changes(edge_scores1, edge_scores2, args.output_dir, top_n=30, show=args.show)
        elif viz_name == 'hard_comparison':
            visualize_hard_circuit_comparison(hard_circuit1, hard_circuit2, edge_scores1, edge_scores2, args.output_dir, args.show)
        elif viz_name == 'circuit_graphs':
            visualize_circuit_graphs(data1, data2, edge_scores1, edge_scores2, hard_circuit1, hard_circuit2, args.output_dir, args.show)
        elif viz_name == 'heatmap':
            visualize_heatmap_comparison(edge_scores1, edge_scores2, hard_circuit1, hard_circuit2, args.output_dir, args.show)
    
    print("\nVisualization generation complete. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()