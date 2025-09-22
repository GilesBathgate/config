"use strict";
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const gridSize = 50;
const cellSize = 10;
canvas.width = gridSize * cellSize;
canvas.height = gridSize * cellSize;
let grid = createGrid();
function createGrid() {
    const grid = [];
    for (let y = 0; y < gridSize; y++) {
        grid[y] = [];
        for (let x = 0; x < gridSize; x++) {
            grid[y][x] = Math.random() > 0.5;
        }
    }
    return grid;
}
function drawGrid() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < gridSize; y++) {
        for (let x = 0; x < gridSize; x++) {
            ctx.beginPath();
            ctx.rect(x * cellSize, y * cellSize, cellSize, cellSize);
            ctx.fillStyle = grid[y][x] ? 'white' : '#121212';
            ctx.fill();
            ctx.strokeStyle = '#333333';
            ctx.stroke();
        }
    }
}
function getNextGeneration() {
    const nextGrid = [];
    for (let y = 0; y < gridSize; y++) {
        nextGrid[y] = [];
        for (let x = 0; x < gridSize; x++) {
            const neighbors = countNeighbors(y, x);
            const isAlive = grid[y][x];
            if (isAlive && (neighbors < 2 || neighbors > 3)) {
                nextGrid[y][x] = false;
            }
            else if (!isAlive && neighbors === 3) {
                nextGrid[y][x] = true;
            }
            else {
                nextGrid[y][x] = isAlive;
            }
        }
    }
    return nextGrid;
}
function countNeighbors(y, x) {
    let count = 0;
    for (let i = -1; i <= 1; i++) {
        for (let j = -1; j <= 1; j++) {
            if (i === 0 && j === 0)
                continue;
            const newY = y + i;
            const newX = x + j;
            if (newY >= 0 && newY < gridSize && newX >= 0 && newX < gridSize) {
                if (grid[newY][newX]) {
                    count++;
                }
            }
        }
    }
    return count;
}
function update() {
    grid = getNextGeneration();
    drawGrid();
    requestAnimationFrame(update);
}
update();
