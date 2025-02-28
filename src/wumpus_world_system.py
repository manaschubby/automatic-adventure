import numpy as np
import random
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from collections import Counter
import queue

class WumpusWorld:
    def __init__(self, size):
        self.size = size
        self.grid = {}  # Use dictionary instead of numpy array for multiple attributes
        for x in range(size):
            for y in range(size):
                self.grid[(x, y)] = {'type': 'Empty', 'percepts': []}

        self.agent_pos = (0, 0)
        self.visited = set()
        self.visit_count = Counter()  # Track how many times each cell has been visited
        self.move_history = []  # Track the last few moves
        self.bayesian_model = None
        self.inference = None
        self.step = 0
        self.knowledge = {}  # Agent's knowledge about the environment
        self.frontier = set()  # Unvisited cells adjacent to visited ones
        self.last_positions = []  # Track the last 5 positions to detect loops
        self.gold_pos = None

        # Randomly place elements and ensure gold is reachable
        self.place_elements_randomly()
        self.build_bayesian_network()

    def place_elements_randomly(self):
        """Randomly place pits, wumpus, and gold while ensuring gold is reachable"""
        # Parameters
        pit_probability = 0.2  # 20% chance of a pit in each cell

        # Clear the grid
        for x in range(self.size):
            for y in range(self.size):
                self.grid[(x, y)] = {'type': 'Empty', 'percepts': []}

        # Keep (0,0) safe as the starting point
        safe_cells = {(0, 0)}

        # Randomly place pits
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in safe_cells:
                    continue  # Skip safe cells

                if random.random() < pit_probability:
                    self.grid[(x, y)]['type'] = 'Pit'

        # Place the Wumpus randomly (but not at the starting point)
        possible_wumpus_locations = [(x, y) for x in range(self.size) for y in range(self.size)
                                  if (x, y) != (0, 0) and self.grid[(x, y)]['type'] == 'Empty']
        if possible_wumpus_locations:
            wumpus_pos = random.choice(possible_wumpus_locations)
            self.grid[wumpus_pos]['type'] = 'Wumpus'

        # Find a location for gold that is reachable from the start
        while True:
            # Place gold randomly (not at the starting point)
            possible_gold_locations = [(x, y) for x in range(self.size) for y in range(self.size)
                                    if (x, y) != (0, 0) and self.grid[(x, y)]['type'] == 'Empty']

            if not possible_gold_locations:
                # Reset the grid if no valid locations for gold
                print("Resetting grid - no valid gold locations")
                return self.place_elements_randomly()

            gold_pos = random.choice(possible_gold_locations)
            self.grid[gold_pos]['type'] = 'Gold'
            self.gold_pos = gold_pos

            # Check if gold is reachable from the start
            if self.is_reachable((0, 0), gold_pos):
                break
            else:
                # Reset and try again if gold is not reachable
                print("Gold not reachable, repositioning...")
                self.grid[gold_pos]['type'] = 'Empty'

        # Add breezes and stenches
        for x in range(self.size):
            for y in range(self.size):
                if self.grid[(x, y)]['type'] == 'Pit':
                    self.add_percept(x, y, 'Breeze')
                elif self.grid[(x, y)]['type'] == 'Wumpus':
                    self.add_percept(x, y, 'Stench')

    def is_reachable(self, start, goal):
        """Check if there's a path from start to goal avoiding pits and wumpus"""
        # Use BFS to find a path
        visited = set()
        q = queue.Queue()
        q.put(start)
        visited.add(start)

        while not q.empty():
            current = q.get()

            if current == goal:
                return True

            for neighbor in self.get_neighbors(*current):
                if (neighbor not in visited and
                    self.grid[neighbor]['type'] != 'Pit' and
                    self.grid[neighbor]['type'] != 'Wumpus'):
                    visited.add(neighbor)
                    q.put(neighbor)

        return False

    def add_percept(self, x, y, percept):
        for nx, ny in self.get_neighbors(x, y):
            if self.grid[(nx, ny)]['type'] == 'Empty' or self.grid[(nx, ny)]['type'] == 'Gold':
                if percept not in self.grid[(nx, ny)]['percepts']:
                    self.grid[(nx, ny)]['percepts'].append(percept)

    def get_neighbors(self, x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbors.append((nx, ny))
        return neighbors

    def build_bayesian_network(self):
        model = BayesianNetwork()

        # Add nodes for pits and wumpus
        for x in range(self.size):
            for y in range(self.size):
                model.add_node(f"Pit_{x}_{y}")
                model.add_node(f"Wumpus_{x}_{y}")

        # Add percept nodes
        for x in range(self.size):
            for y in range(self.size):
                model.add_node(f"Breeze_{x}_{y}")
                model.add_node(f"Stench_{x}_{y}")

        # Add edges
        for x in range(self.size):
            for y in range(self.size):
                # Pits cause breezes in adjacent cells
                pit_node = f"Pit_{x}_{y}"
                for nx, ny in self.get_neighbors(x, y):
                    breeze_node = f"Breeze_{nx}_{ny}"
                    if not model.has_edge(pit_node, breeze_node):
                        model.add_edge(pit_node, breeze_node)

                # Wumpus causes stenches in adjacent cells
                wumpus_node = f"Wumpus_{x}_{y}"
                for nx, ny in self.get_neighbors(x, y):
                    stench_node = f"Stench_{nx}_{ny}"
                    if not model.has_edge(wumpus_node, stench_node):
                        model.add_edge(wumpus_node, stench_node)

        # Add CPDs
        cpds = []

        # Prior probabilities
        for x in range(self.size):
            for y in range(self.size):
                # No pit or wumpus in starting position
                if (x, y) == (0, 0):
                    cpds.append(TabularCPD(f"Pit_{x}_{y}", 2, [[1], [0]]))  # 0% chance of pit
                    cpds.append(TabularCPD(f"Wumpus_{x}_{y}", 2, [[1], [0]]))  # 0% chance of wumpus
                else:
                    cpds.append(TabularCPD(f"Pit_{x}_{y}", 2, [[0.8], [0.2]]))  # 20% chance of pit
                    cpds.append(TabularCPD(f"Wumpus_{x}_{y}", 2, [[0.95], [0.05]]))  # 5% chance of wumpus

        # Breeze CPDs
        for x in range(self.size):
            for y in range(self.size):
                breeze_node = f"Breeze_{x}_{y}"
                neighbors = self.get_neighbors(x, y)

                if not neighbors:
                    cpds.append(TabularCPD(breeze_node, 2, [[1], [0]]))
                    continue

                # A cell has a breeze if ANY adjacent cell has a pit
                evidence = [f"Pit_{nx}_{ny}" for nx, ny in neighbors]
                evidence_card = [2] * len(evidence)

                # Create CPD: P(Breeze=True | Any Pit=True) = 1.0
                num_combinations = 2 ** len(evidence)
                table = []

                for i in range(num_combinations):
                    binary = format(i, f'0{len(evidence)}b')
                    values = [int(bit) for bit in binary]

                    # If any adjacent cell has a pit (1), there's a breeze
                    has_pit = any(v == 1 for v in values)
                    table.append([0 if has_pit else 1, 1 if has_pit else 0])

                cpds.append(TabularCPD(breeze_node, 2, np.array(table).T,
                                      evidence=evidence, evidence_card=evidence_card))

        # Stench CPDs
        for x in range(self.size):
            for y in range(self.size):
                stench_node = f"Stench_{x}_{y}"
                neighbors = self.get_neighbors(x, y)

                if not neighbors:
                    cpds.append(TabularCPD(stench_node, 2, [[1], [0]]))
                    continue

                # A cell has a stench if ANY adjacent cell has a wumpus
                evidence = [f"Wumpus_{nx}_{ny}" for nx, ny in neighbors]
                evidence_card = [2] * len(evidence)

                # Create CPD: P(Stench=True | Any Wumpus=True) = 1.0
                num_combinations = 2 ** len(evidence)
                table = []

                for i in range(num_combinations):
                    binary = format(i, f'0{len(evidence)}b')
                    values = [int(bit) for bit in binary]

                    # If any adjacent cell has a wumpus (1), there's a stench
                    has_wumpus = any(v == 1 for v in values)
                    table.append([0 if has_wumpus else 1, 1 if has_wumpus else 0])

                cpds.append(TabularCPD(stench_node, 2, np.array(table).T,
                                      evidence=evidence, evidence_card=evidence_card))

        for cpd in cpds:
            model.add_cpds(cpd)

        # Check model (this may fail if the model is inconsistent)
        try:
            assert model.check_model()
            self.bayesian_model = model
            self.inference = VariableElimination(model)
        except Exception as e:
            print(f"Error in Bayesian model: {e}")
            # Fallback to simple reasoning if model fails
            self.bayesian_model = None

    def update_knowledge(self):
        """Update agent's knowledge based on current percepts"""
        x, y = self.agent_pos

        # Record what we perceive at the current location
        percepts = self.grid[(x, y)]['percepts']
        self.knowledge[(x, y)] = {
            'breeze': 'Breeze' in percepts,
            'stench': 'Stench' in percepts,
            'safe': True  # If we're here, it's safe
        }

        # Mark all unvisited neighbors as frontier cells (potential next moves)
        for nx, ny in self.get_neighbors(x, y):
            if (nx, ny) not in self.visited:
                self.frontier.add((nx, ny))

            if (nx, ny) not in self.knowledge:
                self.knowledge[(nx, ny)] = {'breeze': False, 'stench': False, 'safe': None}

    def update_frontier(self):
        """Update the frontier set (unvisited cells adjacent to visited cells)"""
        self.frontier = set()
        for x, y in self.visited:
            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in self.visited:
                    self.frontier.add((nx, ny))

    def compute_risk(self):
        """Compute risk map using Bayesian inference or direct reasoning"""
        risk = np.zeros((self.size, self.size))

        # Mark visited cells as safe
        for x, y in self.visited:
            risk[x, y] = 0

        if self.bayesian_model and self.inference:
            # Use Bayesian inference where possible
            evidence = {}

            # Add evidence from visited cells
            for (x, y), info in self.knowledge.items():
                if (x, y) in self.visited:
                    evidence[f"Breeze_{x}_{y}"] = 1 if info['breeze'] else 0
                    evidence[f"Stench_{x}_{y}"] = 1 if info['stench'] else 0

            # Query unvisited cells
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) not in self.visited:
                        try:
                            # Calculate pit risk
                            pit_query = self.inference.query([f"Pit_{x}_{y}"], evidence=evidence)
                            pit_prob = pit_query.values[1]

                            # Calculate wumpus risk
                            wumpus_query = self.inference.query([f"Wumpus_{x}_{y}"], evidence=evidence)
                            wumpus_prob = wumpus_query.values[1]

                            # Combined risk (either pit or wumpus)
                            risk[x, y] = 1 - (1 - pit_prob) * (1 - wumpus_prob)
                        except Exception as e:
                            # Fallback to simple heuristic if inference fails
                            risk[x, y] = self.heuristic_risk(x, y)
        else:
            # Use direct reasoning if no Bayesian model
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) not in self.visited:
                        risk[x, y] = self.heuristic_risk(x, y)

        return risk

    def heuristic_risk(self, x, y):
        """Compute a heuristic risk score for a cell based on adjacent knowledge"""
        if (x, y) in self.visited:
            return 0.0  # Visited cells are safe

        neighbors = self.get_neighbors(x, y)
        visited_neighbors = [n for n in neighbors if n in self.visited]

        if not visited_neighbors:
            return 0.5  # No information about adjacent cells

        breeze_neighbors = [n for n in visited_neighbors if self.knowledge.get(n, {}).get('breeze', False)]
        stench_neighbors = [n for n in visited_neighbors if self.knowledge.get(n, {}).get('stench', False)]

        if breeze_neighbors or stench_neighbors:
            return 0.8  # Dangerous - adjacent to breeze or stench
        else:
            return 0.1  # Likely safe - no breeze or stench in adjacent cells

    def visualize_risk(self, risk):
        """Visualize the risk map"""
        plt.figure(figsize=(8, 8))

        # Create a masked version of the risk matrix for better visualization
        masked_risk = np.copy(risk)
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.visited:
                    pass  # Keep the risk as is
                elif (x, y) == self.agent_pos:
                    masked_risk[x, y] = -1  # Special value for agent
                else:
                    masked_risk[x, y] = -0.5  # Special value for visited cells

        # Custom colormap: agent=blue, visited=green, safe=white to dangerous=red
        cmap = plt.cm.get_cmap('RdYlGn_r').copy()
        cmap.set_under('green')  # Visited cells
        cmap.set_over('blue')    # Agent position

        plt.imshow(masked_risk, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label='Risk Level')

        # Add grid lines
        plt.grid(True, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, self.size, 1), [])
        plt.yticks(np.arange(-.5, self.size, 1), [])

        # Add text annotations
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.visited:
                    if (x, y) == self.agent_pos:
                        plt.text(y, x, 'A', ha='center', va='center', color='white', fontweight='bold')
                    else:
                        breeze = 'B' if self.knowledge.get((x, y), {}).get('breeze', False) else ''
                        stench = 'S' if self.knowledge.get((x, y), {}).get('stench', False) else ''
                        count = self.visit_count[(x, y)]
                        text = f'{breeze}{stench}{"" if count <= 1 else count}'
                        plt.text(y, x, text, ha='center', va='center', color='black')
                else:
                    plt.text(y, x, f'{risk[x, y]:.1f}', ha='center', va='center',
                             color='black' if risk[x, y] < 0.7 else 'white')

        plt.title(f'Wumpus World Risk Map - Step {self.step}')
        plt.savefig(f'wumpus_output/risk_step_{self.step:02d}.png')
        plt.close()

    def is_in_loop(self, next_pos):
        """Check if moving to next_pos would create a loop"""
        if len(self.last_positions) < 4:
            return False

        # Check for oscillation between two positions (A-B-A-B)
        if (next_pos == self.last_positions[-2] and
            self.agent_pos == self.last_positions[-1] and
            next_pos == self.last_positions[-4] and
            self.agent_pos == self.last_positions[-3]):
            return True

        return False

    def best_move(self):
        """Choose the safest adjacent move while avoiding loops"""
        risk = self.compute_risk()
        self.visualize_risk(risk)

        x, y = self.agent_pos
        neighbors = self.get_neighbors(x, y)

        # Calculate a score for each possible move
        move_scores = {}
        for nx, ny in neighbors:
            # Base score is inverse of risk (1 - risk)
            base_score = 1 - risk[nx, ny]

            # Penalize frequently visited cells
            visit_penalty = min(0.5, self.visit_count[(nx, ny)] * 0.1)

            # Extra penalty for moves that would create a loop
            loop_penalty = 0.7 if self.is_in_loop((nx, ny)) else 0

            # Reward for unvisited cells
            novelty_bonus = 0.3 if (nx, ny) not in self.visited else 0

            # Combine factors
            move_scores[(nx, ny)] = base_score - visit_penalty - loop_penalty + novelty_bonus

        # If we have a frontier, prioritize unexplored cells
        frontier_neighbors = [pos for pos in neighbors if pos in self.frontier]
        if frontier_neighbors and any(move_scores[pos] > 0.3 for pos in frontier_neighbors):
            best_frontier = max(frontier_neighbors, key=lambda pos: move_scores[pos])
            return best_frontier

        # If all moves look bad, try a random move
        if all(score < 0.2 for score in move_scores.values()):
            # 30% chance of random move to break out of difficult situations
            if random.random() < 0.3:
                return random.choice(neighbors)

        # Otherwise, pick the best scored move
        return max(move_scores.items(), key=lambda x: x[1])[0]

    def sense_environment(self):
        """Update agent's knowledge about the current location"""
        x, y = self.agent_pos
        cell = self.grid[(x, y)]

        percepts = cell['percepts']
        print(f"Percepts at {self.agent_pos}: {percepts}")

        return {
            'breeze': 'Breeze' in percepts,
            'stench': 'Stench' in percepts,
            'glitter': cell['type'] == 'Gold'
        }

    def print_world(self):
        """Print the current state of the Wumpus World grid"""
        print("\n===== WUMPUS WORLD =====")
        for y in range(self.size-1, -1, -1):  # Print top to bottom
            for x in range(self.size):
                if (x, y) == self.agent_pos:
                    print("A", end=" ")
                else:
                    cell = self.grid[(x, y)]
                    if cell['type'] == 'Empty' and not cell['percepts']:
                        print(".", end=" ")
                    elif cell['type'] == 'Pit':
                        print("P", end=" ")
                    elif cell['type'] == 'Wumpus':
                        print("W", end=" ")
                    elif cell['type'] == 'Gold':
                        print("G", end=" ")
                    elif 'Breeze' in cell['percepts'] and 'Stench' in cell['percepts']:
                        print("BS", end=" ")
                    elif 'Breeze' in cell['percepts']:
                        print("B", end=" ")
                    elif 'Stench' in cell['percepts']:
                        print("S", end=" ")
                    else:
                        print("?", end=" ")
            print()
        print("========================")
        print("A = Agent, P = Pit, W = Wumpus, G = Gold")
        print("B = Breeze, S = Stench, BS = Breeze+Stench")
        print(f"Gold is at position {self.gold_pos}\n")

    def run(self):
        self.print_world()  # Print initial world state

        self.visited.add(self.agent_pos)
        self.visit_count[self.agent_pos] += 1
        self.last_positions.append(self.agent_pos)
        max_steps = self.size * self.size * 3  # Allow more steps

        # Initial sensing
        percepts = self.sense_environment()
        self.update_knowledge()
        self.update_frontier()

        while self.step < max_steps:
            print(f"Step {self.step}: Agent at {self.agent_pos}")
            x, y = self.agent_pos

            # Check for gold
            if self.grid[(x, y)]['type'] == 'Gold':
                print("🎉 Agent found the GOLD at", self.agent_pos)
                break

            # Choose next move
            next_pos = self.best_move()
            print(f"Moving to {next_pos}")

            # Check for revisit
            if next_pos in self.visited:
                visit_count = self.visit_count[next_pos]
                print(f"⚠️ Re-visiting {next_pos} (visit #{visit_count+1})")

            # Update position history to detect loops
            self.last_positions.append(next_pos)
            if len(self.last_positions) > 10:
                self.last_positions.pop(0)

            # Move agent
            nx, ny = next_pos
            self.agent_pos = next_pos

            # Check for death
            if self.grid[next_pos]['type'] in ['Pit', 'Wumpus']:
                print(f"💀 Agent died at {next_pos}! " +
                     ("Fell into a pit." if self.grid[next_pos]['type'] == 'Pit' else "Eaten by Wumpus."))
                self.agent_pos = (0, 0)  # Restart at origin
                self.last_positions = [(0, 0)]  # Reset position history
                self.visit_count[(0, 0)] += 1
                self.visited.add((0, 0))

                # Re-sense the environment after restart
                percepts = self.sense_environment()
                self.update_knowledge()
                self.update_frontier()

                self.step += 1
                continue

            # Update visited, visit count and knowledge
            self.visited.add(next_pos)
            self.visit_count[next_pos] += 1
            percepts = self.sense_environment()
            self.update_knowledge()
            self.update_frontier()

            self.step += 1

            # Check for completion
            if len(self.visited) == self.size * self.size:
                print("🚨 Explored the entire grid. Terminating.")
                break

        else:
            print(f"💥 Max steps ({max_steps}) reached. Terminating.")

        # Final state
        self.print_world()
        print(f"Total steps taken: {self.step}")
        print(f"Unique cells visited: {len(self.visited)} out of {self.size * self.size}")


if __name__ == '__main__':
    size = int(input("Enter grid size (N >= 4): "))
    game = WumpusWorld(size)
    game.print_world()  # Print the world before running
    game.run()
