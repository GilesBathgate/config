import unittest
from main import count_live_neighbors, get_next_generation, ALIVE_CELL, DEAD_CELL

class TestGameOfLife(unittest.TestCase):

    def setUp(self):
        """Set up a reusable grid for testing."""
        self.test_grid = [
            [ALIVE_CELL, ALIVE_CELL, DEAD_CELL],
            [ALIVE_CELL, DEAD_CELL, DEAD_CELL],
            [DEAD_CELL, DEAD_CELL, ALIVE_CELL]
        ]

    def test_count_live_neighbors(self):
        """Test the neighbor counting function."""
        # Cell (x=0, y=1) has 3 neighbors: (0,0), (0,1), (2,2)
        self.assertEqual(count_live_neighbors(self.test_grid, 0, 1), 3)
        # Corner cell (x=0, y=0) has 3 neighbors: (0,1), (1,0), (2,2)
        self.assertEqual(count_live_neighbors(self.test_grid, 0, 0), 3)
        # A dead cell (x=1, y=1) has 4 neighbors: (0,0), (0,1), (1,0), (2,2)
        self.assertEqual(count_live_neighbors(self.test_grid, 1, 1), 4)
        # Corner cell (x=2, y=2) has 3 neighbors: (1,0), (0,1), (0,0)
        self.assertEqual(count_live_neighbors(self.test_grid, 2, 2), 3)


    def test_underpopulation_rule(self):
        """Test Rule 1: A live cell with < 2 neighbors dies."""
        # This grid has a single live cell in the middle, which should die.
        grid = [
            [DEAD_CELL, DEAD_CELL, DEAD_CELL],
            [DEAD_CELL, ALIVE_CELL, DEAD_CELL],
            [DEAD_CELL, DEAD_CELL, DEAD_CELL]
        ]
        next_gen = get_next_generation(grid)
        self.assertEqual(next_gen[1][1], DEAD_CELL)

    def test_survival_rule(self):
        """Test Rule 2: A live cell with 2 or 3 neighbors lives."""
        # The center cell of this "blinker" pattern should survive.
        grid = [
            [DEAD_CELL, ALIVE_CELL, DEAD_CELL],
            [DEAD_CELL, ALIVE_CELL, DEAD_CELL],
            [DEAD_CELL, ALIVE_CELL, DEAD_CELL]
        ]
        next_gen = get_next_generation(grid)
        # Center cell has 2 neighbors, should live
        self.assertEqual(next_gen[1][1], ALIVE_CELL)

    def test_overpopulation_rule(self):
        """Test Rule 3: A live cell with > 3 neighbors dies."""
        # The center cell has 4 neighbors and should die.
        grid = [
            [ALIVE_CELL, DEAD_CELL, ALIVE_CELL],
            [DEAD_CELL, ALIVE_CELL, DEAD_CELL],
            [ALIVE_CELL, DEAD_CELL, ALIVE_CELL]
        ]
        next_gen = get_next_generation(grid)
        self.assertEqual(next_gen[1][1], DEAD_CELL)

    def test_reproduction_rule(self):
        """Test Rule 4: A dead cell with 3 neighbors becomes alive."""
        # The center cell is dead but has 3 neighbors, so it should become alive.
        grid = [
            [ALIVE_CELL, DEAD_CELL, ALIVE_CELL],
            [DEAD_CELL, DEAD_CELL, DEAD_CELL],
            [DEAD_CELL, ALIVE_CELL, DEAD_CELL]
        ]
        next_gen = get_next_generation(grid)
        self.assertEqual(next_gen[1][1], ALIVE_CELL)


if __name__ == '__main__':
    unittest.main()
