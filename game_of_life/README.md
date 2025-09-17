# Conway's Game of Life

This project is a simple, terminal-based implementation of Conway's Game of Life, created by Jules.

## What is it?

Conway's Game of Life is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input. One interacts with the Game of Life by creating an initial configuration and observing how it evolves.

The universe of the Game of Life is an infinite, two-dimensional orthogonal grid of square cells, each of which is in one of two possible states, alive or dead. Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent.

## How to Run the Simulation

To run the simulation, navigate to the `game_of_life` directory and run the `main.py` script:

```bash
python3 main.py
```

Press `Ctrl+C` to stop the simulation.

## How to Run the Tests

To ensure the core logic is working correctly, you can run the included unit tests. Navigate to the `game_of_life` directory and run the `test_game_of_life.py` script:

```bash
python3 -m unittest test_game_of_life.py
```
