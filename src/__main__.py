"""
AI Assignment - Main CLI Interface
"""
import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

from .llm import LLMFactory
from .tic_tac_toe_solver import play_game
from .wumpus_world_system import WumpusWorld
from .integrated_system import IntegratedWumpusTicTacToe
from .utils.plot_binomial_distribution import plot_binomial_distribution

# ----- Constants -----
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_WUMPUS_OUTPUT_DIR = "wumpus_output"
STATE_FILE = "task_state.json"

# ----- Helper Functions -----
def display_opening_screen():
    """Display a fancy opening screen for the CLI"""
    title = r"""
     █████╗ ██╗     █████╗ ███████╗███████╗██╗ ██████╗ ███╗   ██╗███╗   ███╗███████╗███╗   ██╗████████╗
    ██╔══██╗██║    ██╔══██╗██╔════╝██╔════╝██║██╔════╝ ████╗  ██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
    ███████║██║    ███████║███████╗███████╗██║██║  ███╗██╔██╗ ██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║
    ██╔══██║██║    ██╔══██║╚════██║╚════██║██║██║   ██║██║╚██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║
    ██║  ██║██║    ██║  ██║███████║███████║██║╚██████╔╝██║ ╚████║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║
    ╚═╝  ╚═╝╚═╝    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝
    """

    version = "v1.0.0"
    separator = "=" * 80

    print("\033[1;36m" + title + "\033[0m")  # Cyan color for title
    print("\033[1;33m" + f"{version:>80}" + "\033[0m")  # Yellow for version

    # Team information
    print("\033[1;35m" + "Team Members:" + "\033[0m")
    print("\033[1;37m" + "  Shaantanu Jain    - 2021B5A82926H" + "\033[0m")
    print("\033[1;37m" + "  Manas Ashwin      - 2021B4AA1908H" + "\033[0m")
    print("\033[1;37m" + "  Niharika Rao      - 2021B4A32256H" + "\033[0m")
    print("\033[1;37m" + "  Kush Desai        - 2021B4A73158H" + "\033[0m")
    print("\033[1;37m" + "  Utkarsh Bhaskar   - 2021B3A71610H" + "\033[0m")

    print("\033[1;37m" + separator + "\033[0m")  # White for separator

    # Task descriptions
    print("\033[1;32m Available Tasks:\033[0m")
    print(" \033[1;37m1.\033[0m \033[1;34mTic-Tac-Toe Bernoulli Trials\033[0m - Run LLM trials of Tic-Tac-Toe")
    print(" \033[1;37m2.\033[0m \033[1;34mWumpus World Simulation\033[0m     - Navigate a Wumpus World environment")
    print(" \033[1;37m3.\033[0m \033[1;34mIntegrated System\033[0m           - Combined Wumpus-TicTacToe simulation")

    print("\033[1;37m" + separator + "\033[0m")  # White for separator
    print("\033[1;35m Usage: python -m automatic-adventure task1|task2|task3 [options]\033[0m")
    print("\033[1;35m Help:  python -m automatic-adventure --help\033[0m")
    print()


def ensure_output_dirs_exist():
    """Create output directories if they don't exist"""
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEFAULT_WUMPUS_OUTPUT_DIR, exist_ok=True)

def save_state(task: str, state: Dict[str, Any]):
    """Save the current task state to a file"""
    full_state = load_state()
    full_state[task] = state

    with open(os.path.join(DEFAULT_OUTPUT_DIR, STATE_FILE), 'w') as f:
        json.dump(full_state, f, indent=2)

def load_state() -> Dict[str, Any]:
    """Load the saved state from file"""
    try:
        with open(os.path.join(DEFAULT_OUTPUT_DIR, STATE_FILE), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def get_state(task: str) -> Dict[str, Any]:
    """Get the state for a specific task"""
    full_state = load_state()
    return full_state.get(task, {})

def prompt_for_api_key(provider_name: str) -> str:
    """Prompt the user for an API key"""
    key = input(f"Enter your {provider_name} API key: ")
    return key.strip()

def validate_api_key(provider: str, key: str) -> bool:
    """Validate an API key by making a simple request"""
    try:
        llm = LLMFactory.create_provider(provider)
        llm.initialize(api_key=key)
        response = llm.generate("Say 'hello'", temperature=0.1, max_tokens=5)
        print(f"Test response from {provider}: {response.text}")
        return True
    except Exception as e:
        print(f"Error validating {provider} API key: {e}")
        return False

def prompt_int(message: str, min_val: int = 1, max_val: Optional[int] = None, default: Optional[int] = None) -> int:
    """Prompt the user for an integer value within a range"""
    default_str = f" [{default}]" if default is not None else ""
    range_str = f" ({min_val}-{max_val})" if max_val is not None else f" (min: {min_val})"

    while True:
        try:
            value_str = input(f"{message}{range_str}{default_str}: ")
            if not value_str and default is not None:
                return default

            value = int(value_str)
            if value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue

            return value
        except ValueError:
            print("Please enter a valid integer")

def prompt_choice(message: str, choices: List[str], default: Optional[str] = None) -> str:
    """Prompt the user to select from a list of choices"""
    choices_str = "/".join(choices)
    default_str = f" [{default}]" if default is not None else ""

    while True:
        choice = input(f"{message} ({choices_str}){default_str}: ").strip().lower()
        if not choice and default is not None:
            return default

        if choice in [c.lower() for c in choices]:
            return choice

        print(f"Please select one of: {choices_str}")

# ----- Task Runners -----
def run_task1(args):
    """Run Task 1: Tic-Tac-Toe Game with Bernoulli Trials"""
    print("\n=== Task 1: Tic-Tac-Toe Bernoulli Trials ===")

    # Get saved state or initialize new state
    state = get_state("task1")
    outcomes = state.get("outcomes", [])
    total_trials = state.get("total_trials", 0)

    # Check if resuming previous session
    if outcomes and args.continue_previous:
        print(f"Continuing from previous session ({len(outcomes)}/{total_trials} trials completed)")
    else:
        # Reset state
        outcomes = []
        total_trials = 0

    # Get or validate keys
    llm1_key = args.gemini_key or os.getenv("GEMINI_API_KEY") or prompt_for_api_key("Gemini")
    if not validate_api_key("gemini", llm1_key):
        print("Invalid Gemini API key. Exiting.")
        return

    # Default providers
    llm1_provider = "gemini"
    llm2_provider = "gemini"

    # Check if groq key was provided explicitly
    has_explicit_groq_key = args.groq_key is not None

    # If groq key is explicitly provided, use different providers without asking
    if has_explicit_groq_key:
        use_same_key = False
        llm2_provider = "groq"
        llm2_key = args.groq_key

        if not validate_api_key(llm2_provider, llm2_key):
            print(f"Invalid {llm2_provider.capitalize()} API key. Exiting.")
            return
        print("Using Gemini for Player 1 and Groq for Player 2")
    else:
        # Let user choose a second key or use the same
        use_same_key = prompt_choice(
            "Use the same key for both players?", ["y", "n"], default="y"
        ) == "y"

        if use_same_key:
            llm2_key = llm1_key
        else:
            # Ask which provider to use for Player 2
            llm2_provider = prompt_choice(
                "Which provider to use for Player 2?", ["gemini", "groq"], default="groq"
            )

            llm2_key = os.getenv("GROQ_API_KEY") or prompt_for_api_key(llm2_provider.capitalize())

            if not validate_api_key(llm2_provider, llm2_key):
                print(f"Invalid {llm2_provider.capitalize()} API key. Exiting.")
                return

    # Get trial parameters
    if not total_trials:
        board_size = prompt_int("Enter Tic-Tac-Toe board size", min_val=3, max_val=100, default=3)
        total_trials = prompt_int("Enter number of trials to run", min_val=1, default=10)

        # Update state
        state = {
            "outcomes": outcomes,
            "total_trials": total_trials,
            "board_size": board_size,
            "llm1_provider": llm1_provider,
            "llm2_provider": llm2_provider
        }
        save_state("task1", state)
    else:
        board_size = state.get("board_size", 3)
        llm1_provider = state.get("llm1_provider", "gemini")
        llm2_provider = state.get("llm2_provider", "gemini")

    # Calculate remaining trials
    remaining_trials = total_trials - len(outcomes)

    # Run the trials
    try:
        print(f"\nRunning {remaining_trials} Tic-Tac-Toe games (board size: {board_size}x{board_size})...")
        print(f"Player 1: {llm1_provider.capitalize()}, Player 2: {llm2_provider.capitalize()}")

        for i in range(remaining_trials):
            print(f"\n--- Game {len(outcomes) + 1}/{total_trials} ---")
            winner = play_game(
                llm1_key,
                llm2_key,
                board_size,
                llm1_provider=llm1_provider,
                llm2_provider=llm2_provider
            )

            # Record outcome
            outcomes.append(winner)
            state["outcomes"] = outcomes
            save_state("task1", state)

            print(f"Game result: {'Draw' if winner == 0 else f'Player {winner} wins'}")

            # Check if user wants to continue after each game
            if i < remaining_trials - 1 and not args.no_prompt:
                cont = prompt_choice("Continue to next game?", ["y", "n"], default="y")
                if cont != "y":
                    print("Pausing trials. You can continue later using --continue.")
                    break
    except KeyboardInterrupt:
        print("\nTrials interrupted. Progress has been saved.")

    # Skip analysis if no games were played
    if not outcomes:
        print("No games were played. Exiting.")
        return

    # Analyze results
    p1_wins = outcomes.count(1)
    p2_wins = outcomes.count(2)
    draws = outcomes.count(0)

    print("\n=== Results Summary ===")
    print(f"Total games completed: {len(outcomes)}/{total_trials}")
    print(f"Player 1 wins: {p1_wins} ({p1_wins/len(outcomes)*100:.1f}%)")
    print(f"Player 2 wins: {p2_wins} ({p2_wins/len(outcomes)*100:.1f}%)")
    print(f"Draws: {draws} ({draws/len(outcomes)*100:.1f}%)")

    # Save results to file
    result_file = os.path.join(DEFAULT_OUTPUT_DIR, "Exercise1.json")
    with open(result_file, 'w') as f:
        json.dump({
            "total_games": len(outcomes),
            "player1_wins": p1_wins,
            "player2_wins": p2_wins,
            "draws": draws,
            "outcomes": outcomes
        }, f, indent=2)

    print(f"\nResults saved to {result_file}")

    # Generate and save binomial distribution plot
    plot_file = os.path.join(DEFAULT_OUTPUT_DIR, "Exercise1.png")
    plot_binomial_distribution(outcomes, plot_file)
    print(f"Binomial distribution plot saved to {plot_file}")

    # Complete the task if all trials were run
    if len(outcomes) >= total_trials:
        print("All requested trials completed!")

def run_task2(args):
    """Run Task 2: Wumpus World Simulation"""
    print("\n=== Task 2: Wumpus World Simulation ===")

    # Get world size
    world_size = args.size or prompt_int("Enter Wumpus World size", min_val=4, max_val=100, default=4)

    # Get strategy
    strategy_choices = ["best", "random", "mixed"]
    strategy = args.strategy or prompt_choice(
        "Choose movement strategy", strategy_choices, default="best"
    )

    # Create the Wumpus World
    print(f"\nInitializing Wumpus World of size {world_size}x{world_size}...")
    wumpus_world = WumpusWorld(world_size)

    # Print the initial world state
    print("\n=== Initial World State ===")
    wumpus_world.print_world()
    print("Wumpus World created!")

    # Run the simulation
    print(f"\nRunning simulation with '{strategy}' strategy...")
    result = wumpus_world.run_until_completion(strategy=strategy)

    # Print the final world state
    print("\n=== Final World State ===")
    wumpus_world.print_world()

    # Print results
    print("\n=== Simulation Results ===")
    print(f"Found gold: {'Yes' if result['found_gold'] else 'No'}")
    print(f"Steps taken: {result['steps']}")
    print(f"Cells visited: {result['visited_cells']}/{result['total_cells']}")

    # Get and print more detailed state information
    game_state = wumpus_world.get_game_state()

    print("\n=== Detailed State Information ===")
    print(f"Agent final position: {game_state['position']}")
    print(f"Gold position: {game_state['gold_pos']}")
    print(f"Known pit locations: {game_state['known_pits']}")
    print(f"Known wumpus locations: {game_state['known_wumpus']}")

    # Print visit statistics
    print("\n=== Visit Statistics ===")
    visit_counts = wumpus_world.visit_count
    most_visited = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Most visited cells: {most_visited}")

    # Print death statistics if any
    if wumpus_world.deaths:
        print("\n=== Death Statistics ===")
        death_locations = sorted(wumpus_world.deaths.items(), key=lambda x: x[1], reverse=True)
        print(f"Death locations (location: count): {death_locations}")

    # Save results to file
    result_file = os.path.join(DEFAULT_OUTPUT_DIR, "wumpus_results.json")

    # Create a more detailed result dictionary
    detailed_result = {
        **result,  # Include original result data
        "gold_position": game_state['gold_pos'],
        "final_position": game_state['position'],
        "known_pits": list(game_state['known_pits']),
        "known_wumpus": list(game_state['known_wumpus']),
        "most_visited_cells": [{str(loc): count} for loc, count in most_visited],
        "death_locations": [{str(loc): count} for loc, count in wumpus_world.deaths.items()]
    }

    with open(result_file, 'w') as f:
        json.dump(detailed_result, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Risk maps saved to {DEFAULT_WUMPUS_OUTPUT_DIR}/")

    # Print locations of the last few risk map images
    risk_maps = sorted([f for f in os.listdir(DEFAULT_WUMPUS_OUTPUT_DIR)
                       if f.startswith('risk_step_')])[-3:]
    if risk_maps:
        print("\nLast few risk maps:")
        for map_file in risk_maps:
            print(f"  {os.path.join(DEFAULT_WUMPUS_OUTPUT_DIR, map_file)}")

def run_task3(args):
    """Run Task 3: Integrated Wumpus-TicTacToe System"""
    print("\n=== Task 3: Integrated Wumpus-TicTacToe System ===")

    # Get or validate LLM keys (similar to Task 1)
    llm1_key = args.gemini_key or os.getenv("GEMINI_API_KEY") or prompt_for_api_key("Gemini")
    if not validate_api_key("gemini", llm1_key):
        print("Invalid Gemini API key. Exiting.")
        return

    # Default providers
    llm1_provider = "gemini"
    llm2_provider = "gemini"

    # Check if groq key was provided explicitly
    has_explicit_groq_key = args.groq_key is not None

    # If groq key is explicitly provided, use different providers without asking
    if has_explicit_groq_key:
        use_same_key = False
        llm2_provider = "groq"
        llm2_key = args.groq_key

        if not validate_api_key(llm2_provider, llm2_key):
            print(f"Invalid {llm2_provider.capitalize()} API key. Exiting.")
            return
        print("Using Gemini for Player 1 and Groq for Player 2")
    else:
        # Let user choose a second key or use the same
        use_same_key = prompt_choice(
            "Use the same key for both players?", ["y", "n"], default="y"
        ) == "y"

        if use_same_key:
            llm2_key = llm1_key
        else:
            # Ask which provider to use for Player 2
            llm2_provider = prompt_choice(
                "Which provider to use for Player 2?", ["gemini", "groq"], default="groq"
            )

            llm2_key = os.getenv("GROQ_API_KEY") or prompt_for_api_key(llm2_provider.capitalize())

            if not validate_api_key(llm2_provider, llm2_key):
                print(f"Invalid {llm2_provider.capitalize()} API key. Exiting.")
                return

    # Get game parameters
    world_size = args.size or prompt_int("Enter Wumpus World size", min_val=4, max_val=100, default=4)
    ttt_size = prompt_int("Enter Tic-Tac-Toe board size", min_val=3, max_val=100, default=3)
    # Get simulation mode
    mode_choices = ["auto", "step"]
    mode = args.mode or prompt_choice(
        "Choose simulation mode", mode_choices, default="auto"
    )

    # Create the integrated system
    print(f"\nInitializing Wumpus World of size {world_size}x{world_size}...")
    game = IntegratedWumpusTicTacToe(wumpus_size=world_size)

    # Print the initial world state
    print("\n=== Initial Wumpus World State ===")
    game.wumpus_world.print_world()

    # Run the simulation
    if mode == "auto":
        print("\nRunning in automatic mode until completion...")
        # Run the game until completion
        while not game.is_game_over():
            # Play a Tic-Tac-Toe game to determine the move strategy
            ttt_result = game.get_ttt_outcome(
                llm1_key=llm1_key,
                llm2_key=llm2_key,
                board_size=ttt_size,
                llm1_provider=llm1_provider,
                llm2_provider=llm2_provider
            )

            move_strategy = "best" if ttt_result == 0 else "random"
            print(f"\nMove strategy determined: {move_strategy}")

            # Take a step based on the outcome
            result = game.take_step_based_on_ttt_outcome(ttt_result)
            print(f"Move result: {result['message']}")

            if result['status'] == 'found_gold':
                print("🎉 Success! Gold found!")
                break
    else:
        print("\nRunning in step-by-step mode. Press Enter to continue, q to quit.")
        # Run the game step by step with user input
        while not game.is_game_over():
            user_input = input("\nPress Enter to continue, q to quit: ")
            if user_input.lower() == 'q':
                break

            # Play a Tic-Tac-Toe game to determine the move strategy
            ttt_result = game.get_ttt_outcome(
                llm1_key=llm1_key,
                llm2_key=llm2_key,
                board_size=ttt_size,
                llm1_provider=llm1_provider,
                llm2_provider=llm2_provider
            )

            move_strategy = "best" if ttt_result == 0 else "random"
            print(f"\nMove strategy determined: {move_strategy}")

            result = game.take_step_based_on_ttt_outcome(ttt_result)
            print(f"Move result: {result['message']}")
            print(f"Agent position: {result.get('position', 'Unknown')}")

            if result['status'] == 'found_gold':
                print("🎉 Success! Gold found!")
                break

    # Print the final world state
    print("\n=== Final Wumpus World State ===")
    game.wumpus_world.print_world()

    # Game complete - show summary
    summary = game.get_summary()
    print("\nGame Summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    # Save results to file
    result_file = os.path.join(DEFAULT_OUTPUT_DIR, "integrated_results.json")
    with open(result_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_file}")
    print(f"Risk map images saved in '{DEFAULT_WUMPUS_OUTPUT_DIR}' directory")

# ----- Main Function -----
def main():
    """Main entry point for the CLI"""
    # Display the opening screen
    display_opening_screen()

    parser = argparse.ArgumentParser(description="AI Assignment CLI")

    # Create subparsers for each task
    subparsers = parser.add_subparsers(dest="task", help="Task to run")

    # Task 1: Tic-Tac-Toe Bernoulli Trials
    task1_parser = subparsers.add_parser("task1", help="Run Tic-Tac-Toe Bernoulli Trials")
    task1_parser.add_argument("--gemini-key", help="Gemini API key")
    task1_parser.add_argument("--groq-key", help="Groq API key")
    task1_parser.add_argument("--continue", dest="continue_previous", action="store_true",
                             help="Continue from previous state")
    task1_parser.add_argument("--no-prompt", action="store_true",
                             help="Run all trials without prompting")

    # Task 2: Wumpus World Simulation
    task2_parser = subparsers.add_parser("task2", help="Run Wumpus World Simulation")
    task2_parser.add_argument("--size", type=int, help="Size of the Wumpus World grid")
    task2_parser.add_argument("--strategy", choices=["best", "random", "mixed"],
                             help="Movement strategy to use")

    # Task 3: Integrated System
    task3_parser = subparsers.add_parser("task3", help="Run Integrated Wumpus-TicTacToe System")
    task3_parser.add_argument("--gemini-key", help="Gemini API key")
    task3_parser.add_argument("--groq-key", help="Groq API key")
    task3_parser.add_argument("--size", type=int, help="Size of the Wumpus World grid")
    task3_parser.add_argument("--mode", choices=["auto", "step"],
                             help="Run mode: auto or step-by-step")

    # Parse arguments
    args = parser.parse_args()

    # Create output directories
    ensure_output_dirs_exist()

    # Dispatch to the appropriate task
    if args.task == "task1":
        run_task1(args)
    elif args.task == "task2":
        run_task2(args)
    elif args.task == "task3":
        run_task3(args)
    else:
        # If no task specified, show help
        parser.print_help()
        print("\nPlease specify a task to run: task1, task2, or task3")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
