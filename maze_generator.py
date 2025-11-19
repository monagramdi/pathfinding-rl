import numpy as np
from typing import Tuple
import random

class MazeGenerator:
    """Générateur de labyrinthes avec différentes méthodes."""
    
    @staticmethod
    def empty_maze(height: int, width: int) -> np.ndarray:
        """Crée un labyrinthe vide (pas d'obstacles)."""
        return np.zeros((height, width), dtype=int)
    
    @staticmethod
    def simple_maze(height: int, width: int, obstacle_ratio: float = 0.2) -> np.ndarray:
        """
        Crée un labyrinthe simple avec obstacles aléatoires.
        
        Args:
            height: Hauteur de la grille
            width: Largeur de la grille
            obstacle_ratio: Proportion d'obstacles (0-1)
        """
        maze = np.zeros((height, width), dtype=int)
        
        # Ajouter des obstacles aléatoires
        n_obstacles = int(height * width * obstacle_ratio)
        for _ in range(n_obstacles):
            row = random.randint(0, height - 1)
            col = random.randint(0, width - 1)
            maze[row, col] = 1
        
        return maze
    
    @staticmethod
    def corridor_maze(height: int, width: int) -> np.ndarray:
        """Crée un labyrinthe avec des couloirs."""
        maze = np.ones((height, width), dtype=int)
        
        # Créer des couloirs horizontaux et verticaux
        for i in range(1, height, 2):
            maze[i, :] = 0
        
        for j in range(1, width, 2):
            maze[:, j] = 0
        
        return maze
    
    @staticmethod
    def room_maze(height: int, width: int, n_rooms: int = 3) -> np.ndarray:
        """Crée un labyrinthe avec des salles connectées."""
        maze = np.ones((height, width), dtype=int)
        
        rooms = []
        for _ in range(n_rooms):
            room_h = random.randint(3, height // 3)
            room_w = random.randint(3, width // 3)
            room_y = random.randint(1, height - room_h - 1)
            room_x = random.randint(1, width - room_w - 1)
            
            # Créer la salle
            maze[room_y:room_y + room_h, room_x:room_x + room_w] = 0
            rooms.append((room_y + room_h // 2, room_x + room_w // 2))
        
        # Connecter les salles
        for i in range(len(rooms) - 1):
            y1, x1 = rooms[i]
            y2, x2 = rooms[i + 1]
            
            # Corridor horizontal puis vertical
            if x1 < x2:
                maze[y1, x1:x2] = 0
            else:
                maze[y1, x2:x1] = 0
            
            if y1 < y2:
                maze[y1:y2, x2] = 0
            else:
                maze[y2:y1, x2] = 0
        
        return maze
    
    @staticmethod
    def random_walk_maze(height: int, width: int, n_walks: int = 5) -> np.ndarray:
        """Génère un labyrinthe en utilisant des marches aléatoires."""
        maze = np.ones((height, width), dtype=int)
        
        for _ in range(n_walks):
            # Point de départ aléatoire
            y, x = random.randint(0, height - 1), random.randint(0, width - 1)
            maze[y, x] = 0
            
            # Marche aléatoire
            steps = random.randint(height * width // 4, height * width // 2)
            for _ in range(steps):
                direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                new_y = max(0, min(height - 1, y + direction[0]))
                new_x = max(0, min(width - 1, x + direction[1]))
                
                maze[new_y, new_x] = 0
                y, x = new_y, new_x
        
        return maze
    
    @staticmethod
    def ensure_path_exists(maze: np.ndarray, start: Tuple[int, int], 
                          goal: Tuple[int, int]) -> np.ndarray:
        """
        S'assure qu'un chemin existe entre start et goal.
        Utilise BFS pour vérifier, sinon crée un chemin direct.
        """
        maze = maze.copy()
        
        # S'assurer que start et goal sont libres
        maze[start] = 0
        maze[goal] = 0
        
        # Vérifier si un chemin existe (BFS)
        if MazeGenerator._has_path(maze, start, goal):
            return maze
        
        # Créer un chemin simple
        y1, x1 = start
        y2, x2 = goal
        
        # Chemin horizontal puis vertical
        if x1 < x2:
            maze[y1, x1:x2+1] = 0
        else:
            maze[y1, x2:x1+1] = 0
        
        if y1 < y2:
            maze[y1:y2+1, x2] = 0
        else:
            maze[y2:y1+1, x2] = 0
        
        return maze
    
    @staticmethod
    def _has_path(maze: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> bool:
        """Vérifie si un chemin existe entre start et goal (BFS)."""
        from collections import deque
        
        height, width = maze.shape
        visited = set([start])
        queue = deque([start])
        
        while queue:
            y, x = queue.popleft()
            
            if (y, x) == goal:
                return True
            
            # Explorer les voisins
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                
                if (0 <= ny < height and 0 <= nx < width and 
                    maze[ny, nx] == 0 and (ny, nx) not in visited):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
        
        return False
    
    @staticmethod
    def get_random_positions(maze: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Retourne des positions aléatoires valides pour start et goal."""
        height, width = maze.shape
        free_cells = [(i, j) for i in range(height) for j in range(width) 
                      if maze[i, j] == 0]
        
        if len(free_cells) < 2:
            # Si pas assez de cellules libres, forcer start et goal
            return (0, 0), (height - 1, width - 1)
        
        start, goal = random.sample(free_cells, 2)
        return start, goal