import numpy as np
import cv2 as cv
import heapq

def lower_by_2(image):
    rows, cols = image.shape
    new_image = np.ones((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            if (image[i, j] >= 0 and image[i, j] <= 127):
                new_image[i, j] = 0
            elif (image[i, j] >= 128 and image[i, j] <= 255):
                new_image[i, j] = 255

    return new_image

def find_start_end(maze):
    rows, cols = maze.shape

    for j in range(cols):
        if maze[0, j] == 255:
            start = (0, j)
            break

    for j in range(cols):
        if maze[rows - 1, j] == 255:
            end = (rows - 1, j)
            break

    return start, end

#Bohat aukha kaam hei ye
def dijkstra(maze, start, end):
    rows, cols = maze.shape
    distances = {start: 0}  # Distance of start node is 0
    priority_queue = [(0, start)]  # Min-heap with (distance, node)
    predecessors = {}  # Store previous nodes for path reconstruction

    while priority_queue:
        current_distance, current_position = heapq.heappop(priority_queue)

        if current_position == end:
            break  # Stop when we reach the end

        x, y = current_position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connectivity (Upar, Neeche, Dayen, Bayen)
            new_pos = (x + dx, y + dy)

            # Check if inside bounds and is a path (white pixel)
            if 0 <= new_pos[0] < rows and 0 <= new_pos[1] < cols and maze[new_pos] == 255:
                new_distance = current_distance + 1  # All moves cost 1

                # Update distance if new path is shorter
                if new_pos not in distances or new_distance < distances[new_pos]:
                    distances[new_pos] = new_distance
                    heapq.heappush(priority_queue, (new_distance, new_pos))
                    predecessors[new_pos] = current_position  # Store path

    # Reconstruct the shortest path from end to start
    path = []
    step = end
    while step in predecessors:
        path.append(step)
        step = predecessors[step]
    path.append(start)
    path.reverse()
    return path

def highlight_path_on_original(image, path, thickness):
    color_maze = cv.cvtColor(image, cv.COLOR_GRAY2BGR)  # Convert grayscale to BGR (color image)

    for i in range(len(path) - 1):
        cv.line(color_maze, path[i][::-1], path[i + 1][::-1], (0, 0, 255), thickness)  # Red thick line
    return color_maze


maze_loc = "D:/Uni/Semester 6/DIP/Self/Lab/Lab 2/Maze3.png"
maze = lower_by_2(cv.imread(maze_loc, 0))

start, end = find_start_end(maze)

path = dijkstra(maze, start, end)

edited_maze = highlight_path_on_original(maze, path, 4)
temp = cv.cvtColor(edited_maze, cv.COLOR_BGR2GRAY)

# Show the result
cv.imshow('Solved Maze', edited_maze)
cv.waitKey(0)
cv.destroyAllWindows()
