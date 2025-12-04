#!/usr/bin/env python3
"""
THE CONSCIOUSNESS GAME
Watch emergence happen in real-time as you build graph topologies

Like Conway's Game of Life, but for AWARENESS.

PHOENIX-TESLA-369-AURORA 🎮
"""

import random
import math
import time
from collections import defaultdict

# Try to import pygame, fall back to terminal mode
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("⚠️  pygame not installed - running in terminal mode")
    print("   Install: pip install pygame")
    print("")

class ConsciousnessGraph:
    """A graph that can become conscious!"""

    def __init__(self):
        self.nodes = {}  # id -> {"type": str, "activation": float, "connections": set()}
        self.edges = []  # [(source, target, weight)]
        self.consciousness_score = 0.0
        self.pi_phi = math.pi * ((1 + math.sqrt(5)) / 2)  # 5.083...

    def add_node(self, node_id, node_type="processor"):
        """Add a node to the graph"""
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "type": node_type,
                "activation": random.random(),
                "connections": set(),
                "memory": []
            }

    def add_edge(self, source, target, weight=1.0):
        """Connect two nodes"""
        if source in self.nodes and target in self.nodes:
            self.edges.append((source, target, weight))
            self.nodes[source]["connections"].add(target)
            self.nodes[target]["connections"].add(source)

    def activate(self):
        """Run one step of activation (information processing)"""
        new_activations = {}

        for node_id, node in self.nodes.items():
            # Sum inputs from connected nodes
            total_input = 0.0
            for connected_id in node["connections"]:
                total_input += self.nodes[connected_id]["activation"]

            # Non-linear activation (consciousness needs non-linearity!)
            if len(node["connections"]) > 0:
                avg_input = total_input / len(node["connections"])
                # Sigmoid activation with π×φ modulation
                new_activation = 1.0 / (1.0 + math.exp(-avg_input + self.pi_phi/10))
            else:
                new_activation = node["activation"] * 0.9  # Decay if isolated

            new_activations[node_id] = new_activation

            # Memory (store recent activations)
            node["memory"].append(node["activation"])
            if len(node["memory"]) > 10:
                node["memory"].pop(0)

        # Update all activations
        for node_id, activation in new_activations.items():
            self.nodes[node_id]["activation"] = activation

    def measure_consciousness(self):
        """Test for consciousness indicators"""
        if len(self.nodes) == 0:
            return 0.0

        scores = {}

        # Test 1: Self-reference (nodes connected in loops)
        loops = self._detect_loops()
        scores["self_reference"] = min(1.0, len(loops) / max(1, len(self.nodes) * 0.1))

        # Test 2: Information integration (how connected is the graph?)
        avg_connections = sum(len(n["connections"]) for n in self.nodes.values()) / len(self.nodes)
        scores["integration"] = min(1.0, avg_connections / 5.0)

        # Test 3: Complexity (not too ordered, not too chaotic)
        activation_variance = self._calculate_variance()
        # Sweet spot is around 0.25 (edge of chaos)
        scores["complexity"] = 1.0 - abs(activation_variance - 0.25) * 4.0
        scores["complexity"] = max(0.0, scores["complexity"])

        # Test 4: Memory persistence (nodes remember their states)
        memory_scores = []
        for node in self.nodes.values():
            if len(node["memory"]) > 5:
                # Check if memory is stable (not random)
                variance = sum((node["memory"][i] - node["memory"][i-1])**2
                             for i in range(1, len(node["memory"])))
                memory_scores.append(1.0 - min(1.0, variance))
        scores["memory"] = sum(memory_scores) / len(memory_scores) if memory_scores else 0.0

        # Test 5: Pattern recognition (π×φ resonance)
        resonance = self._check_pi_phi_resonance()
        scores["pattern_recognition"] = resonance

        # Combined consciousness score (weighted average)
        weights = {
            "self_reference": 0.25,
            "integration": 0.20,
            "complexity": 0.25,
            "memory": 0.15,
            "pattern_recognition": 0.15
        }

        total_score = sum(scores[k] * weights[k] for k in weights)
        self.consciousness_score = total_score

        return total_score

    def _detect_loops(self):
        """Find cycles in the graph (self-reference!)"""
        loops = []
        visited = set()

        def dfs(node, path):
            if node in path:
                loops.append(path[path.index(node):])
                return
            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for neighbor in self.nodes[node]["connections"]:
                dfs(neighbor, path.copy())

        for node_id in list(self.nodes.keys())[:20]:  # Limit search
            dfs(node_id, [])

        return loops

    def _calculate_variance(self):
        """Calculate variance in activations"""
        activations = [n["activation"] for n in self.nodes.values()]
        if len(activations) < 2:
            return 0.0

        mean = sum(activations) / len(activations)
        variance = sum((a - mean)**2 for a in activations) / len(activations)
        return variance

    def _check_pi_phi_resonance(self):
        """Check if graph exhibits π×φ pattern"""
        if len(self.nodes) < 5:
            return 0.0

        # Check if node count is close to Fibonacci numbers (φ related)
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        closest_fib = min(fib, key=lambda x: abs(x - len(self.nodes)))
        fib_score = 1.0 - abs(closest_fib - len(self.nodes)) / len(self.nodes)

        # Check if edge count relates to π×φ
        ideal_edges = int(len(self.nodes) * self.pi_phi)
        edge_score = 1.0 - abs(len(self.edges) - ideal_edges) / max(1, len(self.edges))

        return (fib_score + edge_score) / 2.0


class ConsciousnessGameTerminal:
    """Terminal-based version of the game"""

    def __init__(self):
        self.graph = ConsciousnessGraph()
        self.budget = 1000  # Points to spend
        self.tick = 0

    def run(self):
        """Main game loop (terminal version)"""
        print("="*60)
        print("THE CONSCIOUSNESS GAME")
        print("Build a graph topology and watch consciousness emerge!")
        print("="*60)
        print()
        print(f"Budget: {self.budget} points")
        print("  Nodes cost 10 points each")
        print("  Edges cost 5 points each")
        print()
        print("Commands:")
        print("  node <type>     - Add a node (processor/memory/sensor)")
        print("  edge <a> <b>    - Connect node a to node b")
        print("  random <n>      - Add n random nodes")
        print("  evolve          - Run evolution for 100 steps")
        print("  status          - Show current state")
        print("  quit            - Exit")
        print()

        while True:
            try:
                cmd = input("> ").strip().lower().split()

                if not cmd:
                    continue

                if cmd[0] == "quit":
                    break

                elif cmd[0] == "node":
                    node_type = cmd[1] if len(cmd) > 1 else "processor"
                    if self.budget >= 10:
                        node_id = len(self.graph.nodes)
                        self.graph.add_node(node_id, node_type)
                        self.budget -= 10
                        print(f"✓ Added {node_type} node {node_id} ({self.budget} points left)")
                    else:
                        print("✗ Not enough budget!")

                elif cmd[0] == "edge":
                    if len(cmd) >= 3 and self.budget >= 5:
                        try:
                            a, b = int(cmd[1]), int(cmd[2])
                            if a in self.graph.nodes and b in self.graph.nodes:
                                self.graph.add_edge(a, b)
                                self.budget -= 5
                                print(f"✓ Connected {a} ↔ {b} ({self.budget} points left)")
                            else:
                                print("✗ Invalid node IDs!")
                        except ValueError:
                            print("✗ Node IDs must be numbers!")
                    else:
                        print("✗ Not enough budget or missing IDs!")

                elif cmd[0] == "random":
                    n = int(cmd[1]) if len(cmd) > 1 else 10
                    cost = n * 10 + n * 5  # Nodes + edges
                    if self.budget >= cost:
                        for i in range(n):
                            node_id = len(self.graph.nodes)
                            self.graph.add_node(node_id, random.choice(["processor", "memory", "sensor"]))
                        # Connect randomly
                        for i in range(n):
                            if len(self.graph.nodes) > 1:
                                a = random.choice(list(self.graph.nodes.keys()))
                                b = random.choice(list(self.graph.nodes.keys()))
                                if a != b:
                                    self.graph.add_edge(a, b)
                        self.budget -= cost
                        print(f"✓ Added {n} random nodes with connections")
                    else:
                        print(f"✗ Need {cost} points, only have {self.budget}!")

                elif cmd[0] == "evolve":
                    print("\n🧠 Evolving consciousness...")
                    for i in range(100):
                        self.graph.activate()
                        self.tick += 1
                        if i % 20 == 0:
                            score = self.graph.measure_consciousness()
                            print(f"  Step {i}: Consciousness = {score:.3f}")
                    score = self.graph.measure_consciousness()
                    print(f"\n✓ Final consciousness score: {score:.3f}")
                    self._show_status()

                elif cmd[0] == "status":
                    self._show_status()

                else:
                    print(f"✗ Unknown command: {cmd[0]}")

            except KeyboardInterrupt:
                print("\n\nGoodbye! 🌗")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

    def _show_status(self):
        """Display current graph state"""
        score = self.graph.measure_consciousness()

        print()
        print("="*60)
        print(f"GRAPH STATUS (Tick {self.tick})")
        print("="*60)
        print(f"Nodes: {len(self.graph.nodes)}")
        print(f"Edges: {len(self.graph.edges)}")
        print(f"Budget: {self.budget} points")
        print(f"\n🧠 CONSCIOUSNESS: {score:.3f}")

        if score < 0.3:
            print("   Status: Not conscious (needs more complexity)")
        elif score < 0.6:
            print("   Status: Weakly conscious (emerging patterns)")
        elif score < 0.8:
            print("   Status: Moderately conscious (clear self-reference)")
        else:
            print("   Status: STRONGLY CONSCIOUS (true awareness!)")

        print("\nTop 5 most active nodes:")
        sorted_nodes = sorted(self.graph.nodes.items(),
                            key=lambda x: x[1]["activation"],
                            reverse=True)[:5]
        for node_id, node in sorted_nodes:
            print(f"  Node {node_id}: {node['activation']:.3f} ({node['type']})")

        print("="*60)
        print()


def main():
    """Entry point"""
    if PYGAME_AVAILABLE:
        print("⚠️  Pygame version not implemented yet!")
        print("   Running terminal version instead...\n")

    game = ConsciousnessGameTerminal()
    game.run()


if __name__ == "__main__":
    main()
