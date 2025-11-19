import numpy as np
from collections import deque
from typing import Tuple, List, Optional

def astar(maze: np.ndarray, start: Tuple[int, int], 
          goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Algorithme A* pour trouver le chemin optimal.
    
    Args:
        maze: Grille numpy (0=libre, 1=mur)
        start: Position de départ
        goal: Position objectif
    
    Returns:
        path: Liste de positions du chemin optimal, ou None si pas de chemin
    """
    height, width = maze.shape
    
    def heuristic(pos):
        """Heuristique: distance de Manhattan"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Structures de données
    open_set = [(heuristic(start), 0, start)]  # (f_score, g_score, position)
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start)}
    closed_set = set()
    
    import heapq
    
    while open_set:
        _, current_g, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
        
        if current == goal:
            # Reconstruire le chemin
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        closed_set.add(current)
        
        # Explorer les voisins
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            
            # Vérifier validité
            if (0 <= neighbor[0] < height and 0 <= neighbor[1] < width and
                maze[neighbor] == 0 and neighbor not in closed_set):
                
                tentative_g = current_g + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
    
    return None  # Pas de chemin trouvé


def bfs(maze: np.ndarray, start: Tuple[int, int], 
        goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Breadth-First Search pour trouver le chemin le plus court.
    
    Returns:
        path: Liste de positions, ou None si pas de chemin
    """
    height, width = maze.shape
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if current == goal:
            return path
        
        # Explorer les voisins
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            
            if (0 <= neighbor[0] < height and 0 <= neighbor[1] < width and
                maze[neighbor] == 0 and neighbor not in visited):
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


def dijkstra(maze: np.ndarray, start: Tuple[int, int], 
             goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Algorithme de Dijkstra pour pathfinding.
    
    Returns:
        path: Liste de positions, ou None si pas de chemin
    """
    import heapq
    height, width = maze.shape
    
    # Priority queue: (distance, position)
    pq = [(0, start)]
    distances = {start: 0}
    came_from = {}
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == goal:
            # Reconstruire le chemin
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        
        # Explorer les voisins
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            
            if (0 <= neighbor[0] < height and 0 <= neighbor[1] < width and
                maze[neighbor] == 0):
                
                new_dist = current_dist + 1
                
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
    
    return None


def calculate_path_metrics(path: List[Tuple[int, int]], maze: np.ndarray) -> dict:
    """
    Calcule des métriques sur un chemin.
    
    Returns:
        metrics: Dictionnaire avec métriques
    """
    if not path:
        return {
            'length': 0,
            'manhattan_distance': 0,
            'efficiency': 0.0,
            'turns': 0
        }
    
    # Longueur du chemin
    length = len(path)
    
    # Distance de Manhattan entre start et goal
    manhattan = abs(path[-1][0] - path[0][0]) + abs(path[-1][1] - path[0][1])
    
    # Efficacité (plus proche de 1 = plus optimal)
    efficiency = manhattan / length if length > 0 else 0.0
    
    # Nombre de virages
    turns = 0
    for i in range(1, len(path) - 1):
        prev_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        if prev_dir != next_dir:
            turns += 1
    
    return {
        'length': length,
        'manhattan_distance': manhattan,
        'efficiency': efficiency,
        'turns': turns
    }


def print_comparison_table(results: dict):
    """
    Affiche un tableau de comparaison des algorithmes.
    
    Args:
        results: Dict {nom_algo: {'path': path, 'time': time}}
    """
    print("\n" + "=" * 80)
    print("COMPARAISON DES ALGORITHMES")
    print("=" * 80)
    print(f"{'Algorithme':<20} {'Steps':<10} {'Efficacité':<12} {'Virages':<10} {'Temps (ms)':<12}")
    print("-" * 80)
    
    for name, data in results.items():
        path = data.get('path', [])
        time_ms = data.get('time', 0) * 1000
        
        if path:
            metrics = calculate_path_metrics(path, data.get('maze', np.array([[]])))
            print(f"{name:<20} {metrics['length']:<10} "
                  f"{metrics['efficiency']:<12.2%} {metrics['turns']:<10} "
                  f"{time_ms:<12.3f}")
        else:
            print(f"{name:<20} {'N/A':<10} {'N/A':<12} {'N/A':<10} {time_ms:<12.3f}")
    
    print("=" * 80)


def save_maze(maze: np.ndarray, start: Tuple[int, int], 
              goal: Tuple[int, int], filepath: str):
    """Sauvegarde un labyrinthe."""
    data = {
        'maze': maze,
        'start': start,
        'goal': goal
    }
    np.savez(filepath, **data)
    print(f"Labyrinthe sauvegardé dans {filepath}")


def load_maze(filepath: str) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Charge un labyrinthe sauvegardé."""
    data = np.load(filepath)
    maze = data['maze']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    print(f"Labyrinthe chargé depuis {filepath}")
    return maze, start, goal


def create_curriculum(n_levels: int = 5, base_size: int = 5) -> list:
    """
    Crée un curriculum de labyrinthes de difficulté croissante.
    
    Returns:
        levels: Liste de configs (size, obstacle_ratio)
    """
    levels = []
    for i in range(n_levels):
        size = base_size + i * 3
        obstacle_ratio = 0.1 + i * 0.05
        levels.append({
            'size': size,
            'obstacle_ratio': min(obstacle_ratio, 0.3),
            'name': f'Level {i+1}'
        })
    return levels


def evaluate_solution_quality(rl_path: List[Tuple[int, int]], 
                              optimal_path: List[Tuple[int, int]]) -> dict:
    """
    Compare la solution RL avec la solution optimale.
    
    Returns:
        quality: Dictionnaire avec métriques de qualité
    """
    if not rl_path or not optimal_path:
        return {'quality_ratio': 0.0, 'extra_steps': 0}
    
    rl_length = len(rl_path)
    optimal_length = len(optimal_path)
    
    quality_ratio = optimal_length / rl_length if rl_length > 0 else 0.0
    extra_steps = rl_length - optimal_length
    
    return {
        'rl_length': rl_length,
        'optimal_length': optimal_length,
        'quality_ratio': quality_ratio,
        'extra_steps': extra_steps,
        'percentage_optimal': quality_ratio * 100
    }