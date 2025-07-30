import os
import json
import networkx as nx
from jsonschema import validate, ValidationError
import string
import argparse
# TODO: Uncomment the following import once compute_stability.py is implemented
# from ..compute_stability import get_stability_score, get_response, get_circuit_data
try:
    from ..compute_stability import get_stability_score, get_response, get_circuit_data
except ImportError:
    # Dummy functions for testing until compute_stability.py is implemented
    def get_stability_score(variation): return 0.5
    def get_response(variation): return ""
    def get_circuit_data(variation): return f"dummy_path/{variation.replace(' ', '_')}"

##################################################################################
# GitHub Issue #21: Storing circuit data in JSON format
# This schema defines the structure for storing thought variations and their
# circuit data in a standardized tree format for easy serialization/deserialization.
# It ensures data integrity when saving and loading experiments.
##################################################################################
TREE_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "current_depth": {"type": "integer"},
                    "text": {"type": "string"},
                    "children": {"type": "array", "items": {"type": "string"}},
                    "circuit_data": {"type": "string"},
                    "score": {"type": "number"}
                },
                "required": ["id", "current_depth", "text", "children", "circuit_data", "score"]
            }
        }
    },
    "required": ["nodes"]
}

##################################################################################
# The CircuitTree class implements a tree structure to organize and manage 
# thought variations and their circuit data. This implementation follows a sequential
# development plan addressing GitHub issues #11, #21, and #22.
##################################################################################
class CircuitTree:
    # Constructor to initialize the tree structure using NetworkX's DiGraph
    def __init__(self, output_dir='outputs', root_prefix='root'):
        """
        Initializes the CircuitTree with a NetworkX DiGraph, output directory,
        and a prefix for the root node.
        """
        self.tree = nx.DiGraph() # Using DiGraph to maintain directed edges
        self.output_dir = output_dir # Directory where experiment JSON files will be saved
        os.makedirs(self.output_dir, exist_ok=True) # Ensure output directory exists
        self.branch_counters = {} # Dictionary to keep track of branch indices for unique node IDs
        self.root_prefix = root_prefix # Prefix for the root node ID, used to generate unique IDs for child nodes

    ##################################################################################
    # GitHub Issue #11: Retrieving circuit data and stability scores
    # This method implements the integration with compute_stability.py to get
    # stability scores and circuit data for thought variations. This is the first
    # key functionality needed to build our tree structure.
    ##################################################################################
    def add_variation_from_stability_data(self, variation, parent_id=None, level=0):
        """
        Add a node to the tree using data retrieved from compute_stability.py.
        
        This method serves as the integration point between the CircuitTree structure
        and the circuit stability computation system. It calls getter functions from
        compute_stability.py to obtain stability scores and circuit data for a 
        given thought variation.
        
        Parameters:
        -----------
        variation : str
            The thought variation text to be evaluated
        parent_id : str, optional
            ID of the parent node (default is None for root node)
        level : int, optional
            Depth level in the tree (default is 0)
            
        Returns:
        --------
        str
            ID of the newly created node
            
        Note:
        -----
        This function requires compute_stability.py to implement:
        - get_stability_score(variation): Returns a numerical score
        - get_response(variation): Returns the LLM response
        - get_circuit_data(variation): Returns path to circuit data
        """
        # TODO: Uncomment and use these functions once compute_stability.py is implemented
        # Retrieve stability score for this variation
        # score = get_stability_score(variation)
        
        # Retrieve LLM response for this variation (may be used for debugging/display)
        # response = get_response(variation)
        
        # Retrieve circuit data path for this variation
        # circuit_data = get_circuit_data(variation)
        
        # For now, use dummy values until the actual functions are implemented
        score = 0.5  # Dummy score
        circuit_data = f"dummy_path/{variation.replace(' ', '_')}"  # Dummy circuit data path
        
        # Add node to the tree with the retrieved data
        return self.add_node(variation, score, circuit_data, parent_id, level)
    ##################################################################################
    # END: GitHub Issue #11: Retrieving circuit data and stability scores
    ##################################################################################

    ##################################################################################
    # GitHub Issue #21: Storing circuit data in JSON format
    # These methods implement the core tree structure and node management functionality.
    # This builds upon Issue #11 by providing a structure to store and organize the 
    # circuit data and stability scores.
    ##################################################################################
    # Logic for adding nodes to the tree
    def add_node(self, variation, score, circuit_data, parent_id=None, level=0):
        """
        Adds a single node to the tree. Each node represents a thought variation
        and stores its score and circuit data. It also handles node ID generation
        based on parent-child relationships.
        """
        # Error handling for input validation
        if not isinstance(variation, str):
            raise ValueError("variation must be a string")
        if not isinstance(score, (int, float)):
            raise ValueError("score must be a number")
        if not isinstance(circuit_data, str):
            raise ValueError("circuit_data must be a string path")
            
        if parent_id is None:
            # This is the root node of the tree
            node_id = self.root_prefix
            self.branch_counters[node_id] = 0
        else:
            # Generate a unique ID for child nodes
            branch_index = self.branch_counters.get(parent_id, 0)
            branch_label = string.ascii_uppercase[branch_index % 26]
            node_id = f"{parent_id}-{branch_label}"
            self.branch_counters[parent_id] = branch_index + 1
            self.branch_counters[node_id] = 0
        
        self.tree.add_node(node_id, text=variation, score=score, circuit_data=circuit_data, 
                           current_depth=level + 1, children=[])
        
        if parent_id:
            # Connect the new node to its parent
            self.tree.add_edge(parent_id, node_id)
            self.tree.nodes[parent_id]['children'].append(node_id)
        
        return node_id

    def add_nodes_batch(self, nodes_data):
        """
        Adds multiple nodes to the tree from a list of node data.
        Useful for bulk operations when importing multiple thought variations.
        """
        node_ids = []
        for node_data in nodes_data:
            node_id = self.add_node(**node_data)
            node_ids.append(node_id)
        return node_ids
    ##################################################################################
    # END: Core tree node management functionality
    ##################################################################################

    ##################################################################################
    # GitHub Issue #22: Selecting best plan and backtracking based on circuit stability scores
    # These methods implement functionality to navigate the tree, find optimal nodes,
    # and enable backtracking when a path fails. This builds upon issues #11 and #21
    # by providing ways to navigate and utilize the tree structure.
    ##################################################################################
    def get_best_at_level(self, level):
        """
        Finds the node with the highest score at a specific level of the tree.
        This is used to select the most promising thought variation to expand upon.
        """
        level_nodes = [n for n, d in self.tree.nodes(data=True) if d['current_depth'] == level + 1]
        if not level_nodes:
            return None
        return max(level_nodes, key=lambda n: self.tree.nodes[n]['score'])

    def get_nodes_at_level(self, level, sort_by_score=True):
        """
        Retrieves all nodes at a given level, optionally sorted by score.
        This allows for considering second-best options if the best one fails.
        """
        level_nodes = [n for n, d in self.tree.nodes(data=True) if d['current_depth'] == level + 1]
        if sort_by_score:
            return sorted(level_nodes, key=lambda n: self.tree.nodes[n]['score'], reverse=True)
        return level_nodes

    def backtrack(self, node_id):
        """
        Finds the parent of a given node, allowing the experiment to backtrack
        to a previous step if the current path is not fruitful.
        """
        predecessors = list(self.tree.predecessors(node_id))
        return predecessors[0] if predecessors else None

    def get_parent_from_id(self, node_id):
        """
        Helper function to determine a node's parent from its ID string.
        Useful for navigation and backtracking in the tree structure.
        """
        if '-' not in node_id:
            return None
        return node_id.rsplit('-', 1)[0]
    ##################################################################################
    # END: GitHub Issue #22: Selecting best plan and backtracking
    ##################################################################################
    
    ##################################################################################
    # GitHub Issue #21: Serialization and deserialization functionality
    # These methods handle saving and loading tree structures to/from JSON files,
    # completing the data persistence requirements from Issue #21.
    ##################################################################################
    def save_experiment_json(self, experiment_id):
        """
        Saves the current tree structure to a JSON file. It validates the data
        against TREE_SCHEMA before saving to ensure data integrity.
        """
        tree_data = {"nodes": []}
        for node, data in self.tree.nodes(data=True):
            tree_data["nodes"].append({
                "id": node,
                "current_depth": data['current_depth'],
                "text": data['text'],
                "children": data['children'],
                "circuit_data": data['circuit_data'],
                "score": data['score']
            })
        
        try:
            # Ensure the data conforms to our schema before saving
            validate(instance=tree_data, schema=TREE_SCHEMA)
        except ValidationError as e:
            raise ValueError(f"JSON validation failed: {e}")
        
        # Create the full file path and save the tree data
        file_path = os.path.join(self.output_dir, f"{experiment_id}.json")
        with open(file_path, 'w') as f:
            json.dump(tree_data, f, indent=4)  # Pretty print with indentation
        print(f"Saved tree for experiment '{experiment_id}' to {file_path}")
        return file_path

    def load_experiment_json(self, file_path):
        """
        Loads a tree structure from a JSON file, validates it, and reconstructs
        the NetworkX DiGraph.
        """
        # Read the JSON file
        with open(file_path, 'r') as f:
            tree_data = json.load(f)
        
        try:
            # Ensure the loaded data conforms to our schema
            validate(instance=tree_data, schema=TREE_SCHEMA)
        except ValidationError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Reset the tree and branch counters
        self.tree = nx.DiGraph()
        self.branch_counters = {}
        
        # First pass: Recreate all nodes
        for node_data in tree_data["nodes"]:
            self.tree.add_node(
                node_data["id"],
                text=node_data["text"],
                score=node_data["score"],
                circuit_data=node_data["circuit_data"],
                current_depth=node_data["current_depth"],
                children=node_data["children"]
            )
        
        # Second pass: Recreate all edges between nodes
        for node_data in tree_data["nodes"]:
            for child_id in node_data["children"]:
                self.tree.add_edge(node_data["id"], child_id)
        
        # Third pass: Restore branch counters for proper node ID generation in future additions
        for node_id in self.tree.nodes():
            parent_id = self.get_parent_from_id(node_id)
            if parent_id:
                branch_label = node_id.split('-')[-1]
                if len(branch_label) == 1 and branch_label.isalpha():
                    branch_index = ord(branch_label) - ord('A') + 1
                    self.branch_counters[parent_id] = max(
                        self.branch_counters.get(parent_id, 0),
                        branch_index
                    )
        
        print(f"Loaded tree from {file_path} with {len(self.tree.nodes)} nodes")
        return True
    ##################################################################################
    # END: GitHub Issue #21: Serialization and deserialization functionality
    ##################################################################################

    ##################################################################################
    # Test helpers - Used for development and validation of the tree structure.
    ##################################################################################
    def create_test_tree(self, root_variation="Root thought", levels=2, branches_per_level=2):
        """
        Generates a sample tree for testing and demonstration purposes.
        Creates a multi-level tree with specified branching factor for visualization
        and testing of tree functionality.
        """
        # Create the root node of the test tree
        root_id = self.add_node(
            variation=root_variation,
            score=0.9,  # Root has highest score
            circuit_data="circuits/test/root/",
            parent_id=None,
            level=0
        )
        
        # Recursive helper function to build the tree structure
        def add_children(parent_id, current_level, max_level):
            # Base case: stop recursion when max depth is reached
            if current_level >= max_level:
                return
            
            # Create branches at this level
            for i in range(branches_per_level):
                # Create a score that varies slightly across branches
                score = 0.8 + (0.1 * i / branches_per_level)
                
                # Create the child node
                child_id = self.add_node(
                    variation=f"Level {current_level+1} Branch {i+1}",
                    score=score,
                    circuit_data=f"circuits/test/level{current_level+1}_branch{i+1}/",
                    parent_id=parent_id,
                    level=current_level+1
                )
                
                # Recursively add children to this node
                add_children(child_id, current_level+1, max_level)
        
        # Start building the tree from the root
        add_children(root_id, 0, levels)
        
        return root_id
    ##################################################################################
    # END: Test helpers
    ##################################################################################

##################################################################################
# Command-line interface for testing and demonstrating tree functionality.
# This allows running the script from the command line to test different features
# across all implemented GitHub issues (#11, #21, #22).
##################################################################################
def main():
    """
    Command line interface that demonstrates the functionality of CircuitTree across
    all GitHub issues (#11, #21, #22).
    
    This function parses command line arguments and runs the appropriate CircuitTree
    operations based on those arguments, enabling testing of different features:
    - Issue #11: Test stability integration with --test_stability
    - Issue #21: Test serialization with --experiment_id and --input_json
    - Issue #22: Navigation and tree structure demonstrated with --test
    """
    parser = argparse.ArgumentParser(description="Run CircuitTree with inputs for tree building and saving.")
    # Basic configuration arguments
    parser.add_argument("--experiment_id", type=str, required=True, help="ID for the experiment JSON file (required).")
    parser.add_argument("--root_prefix", type=str, default="root", help="Prefix for root node ID.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output files.")
    
    # GitHub Issue #21 related arguments
    parser.add_argument("--input_json", type=str, help="Path to input JSON with node data for adding multiple nodes (format: list of dicts with 'variation', 'score', 'circuit_data', 'parent_id', 'level').")
    
    # Test structure generation arguments
    parser.add_argument("--test", action="store_true", help="Run in test mode to create a sample tree.")
    parser.add_argument("--test_levels", type=int, default=2, help="Number of levels for test tree.")
    parser.add_argument("--test_branches", type=int, default=2, help="Branches per level for test tree.")
    
    # GitHub Issue #11 integration test arguments
    parser.add_argument("--test_stability", action="store_true", help="Test the integration with stability data.")
    parser.add_argument("--variation", type=str, help="Test thought variation to evaluate with stability functions.")
    
    args = parser.parse_args()
    
    # Initialize the tree structure
    tree = CircuitTree(output_dir=args.output_dir, root_prefix=args.root_prefix)
    
    if args.test:
        # Test the tree structure creation (GitHub Issue #21 & #22)
        print(f"Creating test tree with {args.test_levels} levels and {args.test_branches} branches per level")
        tree.create_test_tree(levels=args.test_levels, branches_per_level=args.test_branches)
    elif args.test_stability:
        # Test the integration with compute_stability.py (GitHub Issue #11)
        if args.variation:
            print(f"Testing stability integration with variation: '{args.variation}'")
            node_id = tree.add_variation_from_stability_data(args.variation)
            print(f"Added node with ID: {node_id}")
        else:
            # Add some sample variations if none provided
            variations = [
                "This is the first thought variation",
                "This is a different approach to the problem",
                "Let's try a completely new direction"
            ]
            for variation in variations:
                node_id = tree.add_variation_from_stability_data(variation)
                print(f"Added variation '{variation}' with node ID: {node_id}")
    elif args.input_json:
        # Test batch node addition from JSON (GitHub Issue #21)
        with open(args.input_json, 'r') as f:
            nodes_data = json.load(f)
        if isinstance(nodes_data, list):
            node_ids = tree.add_nodes_batch(nodes_data)
            print(f"Added {len(node_ids)} nodes from JSON input")
        else:
            print("Warning: input_json should contain a list of node data dictionaries")
    else:
        print("No input_json provided; add nodes manually or via pipeline.")
    
    # Save the experiment - Tests GitHub Issue #21 serialization
    file_path = tree.save_experiment_json(args.experiment_id)
    print(f"Tree saved to {file_path}")

if __name__ == "__main__":
    main() 