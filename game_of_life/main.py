import random
import time

# --- Constants ---
GRID_WIDTH = 40
GRID_HEIGHT = 20
INITIAL_POPULATION_DENSITY = 0.25  # 25% of cells will be alive initially
SIMULATION_SPEED = 0.1  # Seconds between each generation

# --- Cell Representation ---
ALIVE_CELL = "■"
DEAD_CELL = " "


def create_initial_grid(width, height):
    """Creates a grid with a random initial population of cells."""
    grid = []
    for _ in range(height):
        row = []
        for _ in range(width):
            if random.random() < INITIAL_POPULATION_DENSITY:
                row.append(ALIVE_CELL)
            else:
                row.append(DEAD_CELL)
        grid.append(row)
    return grid


def print_grid(grid):
    """Prints the entire grid to the console."""
    # Clear the console screen using an ANSI escape code for security.
    print("\033[H\033[J", end="")
    print("--- Conway's Game of Life ---")
    for row in grid:
        print(" ".join(row))
    print("-----------------------------")
    print("Press Ctrl+C to exit.")


def count_live_neighbors(grid, x, y):
    """Counts the number of live neighbors for a given cell."""
    live_neighbors = 0
    height = len(grid)
    width = len(grid[0])

    # Iterate through the 8 neighbors
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Skip the cell itself
            if i == 0 and j == 0:
                continue

            # Calculate neighbor coordinates with wrap-around (toroidal array)
            neighbor_y = (y + i) % height
            neighbor_x = (x + j) % width

            if grid[neighbor_y][neighbor_x] == ALIVE_CELL:
                live_neighbors += 1

    return live_neighbors


def get_next_generation(current_grid):
    """Calculates the next generation of the grid based on Conway's rules."""
    height = len(current_grid)
    width = len(current_grid[0])
    next_grid = [[DEAD_CELL for _ in range(width)] for _ in range(height)]

    for y in range(height):
        for x in range(width):
            live_neighbors = count_live_neighbors(current_grid, x, y)
            cell = current_grid[y][x]

            # Rule 1: A live cell with fewer than two live neighbours dies (underpopulation).
            if cell == ALIVE_CELL and live_neighbors < 2:
                next_grid[y][x] = DEAD_CELL
            # Rule 2: A live cell with two or three live neighbours lives on to the next generation.
            elif cell == ALIVE_CELL and (live_neighbors == 2 or live_neighbors == 3):
                next_grid[y][x] = ALIVE_CELL
            # Rule 3: A live cell with more than three live neighbours dies (overpopulation).
            elif cell == ALIVE_CELL and live_neighbors > 3:
                next_grid[y][x] = DEAD_CELL
            # Rule 4: A dead cell with exactly three live neighbours becomes a live cell (reproduction).
            elif cell == DEAD_CELL and live_neighbors == 3:
                next_grid[y][x] = ALIVE_CELL

    return next_grid


def main():
    """Main function to run the simulation."""
    grid = create_initial_grid(GRID_WIDTH, GRID_HEIGHT)

    try:
        while True:
            print_grid(grid)
            grid = get_next_generation(grid)
            time.sleep(SIMULATION_SPEED)
    except KeyboardInterrupt:
        print("\nSimulation ended. Thanks for playing!")


if __name__ == "__main__":
    main()
