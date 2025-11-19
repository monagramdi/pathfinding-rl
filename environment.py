import numpy as np
from typing import Tuple, Optional

class MazeEnvironment:
    """
    Environnement de labyrinthe pour l'apprentissage par renforcement.
    """
    
    def __init__(self, maze: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """
        Args:
            maze: Grille numpy (0=libre, 1=mur)
            start: Position de départ (row, col)
            goal: Position objectif (row, col)
        """
        self.maze = maze.copy()
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.height, self.width = maze.shape
        
        # Actions: 0=Haut, 1=Bas, 2=Gauche, 3=Droite
        self.actions = {
            0: (-1, 0),  # Haut
            1: (1, 0),   # Bas
            2: (0, -1),  # Gauche
            3: (0, 1)    # Droite
        }
        self.n_actions = len(self.actions)
        
        # Statistiques
        self.steps = 0
        self.max_steps = min(self.height * self.width * 2, 500)  # Limite de sécurité
        self.visited = []  # Liste pour tracker l'ordre
        self.visited_set = set()  # Set pour vérifications rapides
        self.consecutive_collisions = 0  # Compteur de collisions
        
    def reset(self) -> Tuple[int, int]:
        """Réinitialise l'environnement."""
        self.current_pos = self.start
        self.steps = 0
        self.visited = [self.start]
        self.visited_set = {self.start}
        self.consecutive_collisions = 0
        return self.current_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        Exécute une action.
        
        Returns:
            next_state: Nouvelle position
            reward: Récompense obtenue
            done: Episode terminé?
            info: Informations additionnelles
        """
        self.steps += 1
        
        # Calculer nouvelle position
        delta = self.actions[action]
        new_row = self.current_pos[0] + delta[0]
        new_col = self.current_pos[1] + delta[1]
        new_pos = (new_row, new_col)
        
        # Vérifier si la nouvelle position est valide
        if not self._is_valid_position(new_pos):
            # Collision avec mur ou hors limites - on ne bouge pas
            reward = -10
            # Compter les collisions consécutives
            if not hasattr(self, 'consecutive_collisions'):
                self.consecutive_collisions = 0
            self.consecutive_collisions += 1
            
            # Si trop de collisions consécutives, arrêter l'épisode
            if self.consecutive_collisions > 10:
                done = True
                info = {'reason': 'too_many_collisions', 'steps': self.steps}
                return self.current_pos, -50, done, info
            
            done = False
            info = {'reason': 'collision'}
            return self.current_pos, reward, done, info
        
        # Réinitialiser le compteur de collisions si mouvement valide
        self.consecutive_collisions = 0
        
        # Déplacer l'agent
        self.current_pos = new_pos
        self.visited.append(new_pos)
        self.visited_set.add(new_pos)
        
        # Vérifier si objectif atteint
        if self.current_pos == self.goal:
            reward = 100
            done = True
            info = {'reason': 'goal_reached', 'steps': self.steps}
            return self.current_pos, reward, done, info
        
        # Vérifier limite de steps
        if self.steps >= self.max_steps:
            reward = -50
            done = True
            info = {'reason': 'max_steps', 'steps': self.steps}
            return self.current_pos, reward, done, info
        
        # Détection de boucle infinie (même position visitée trop souvent)
        if len(self.visited) > 10:
            # Compter combien de fois la position actuelle apparaît dans les 20 derniers steps
            recent_visits = self.visited[-20:]
            visit_count = recent_visits.count(self.current_pos)
            if visit_count > 4:  # Retour trop fréquent à la même position = boucle
                reward = -50
                done = True
                info = {'reason': 'stuck_in_loop', 'steps': self.steps}
                return self.current_pos, reward, done, info
        
        # Récompense standard: pénalité pour chaque step + bonus si se rapproche
        distance_before = self._manhattan_distance(
            (self.current_pos[0] - delta[0], self.current_pos[1] - delta[1]), 
            self.goal
        )
        distance_after = self._manhattan_distance(self.current_pos, self.goal)
        
        # Reward shaping: bonus si se rapproche, pénalité si s'éloigne
        if distance_after < distance_before:
            reward = -1 + 0.5  # Se rapproche
        elif new_pos in self.visited_set and len(self.visited) > 5:
            # Pénalité plus forte pour revisiter une case
            reward = -1 - 0.5
        else:
            reward = -1  # Case nouvelle mais pas forcément la bonne direction
        
        done = False
        info = {'reason': 'continue', 'distance_to_goal': distance_after}
        
        return self.current_pos, reward, done, info
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Vérifie si une position est valide."""
        row, col = pos
        
        # Hors limites
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        
        # Mur
        if self.maze[row, col] == 1:
            return False
        
        return True
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calcule la distance de Manhattan entre deux positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_state(self) -> Tuple[int, int]:
        """Retourne l'état actuel."""
        return self.current_pos
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """Affiche l'environnement."""
        display = self.maze.copy().astype(str)
        display[display == '0'] = '.'
        display[display == '1'] = '#'
        
        # Marquer les positions visitées (sauf les plus récentes)
        if len(self.visited) > 5:
            for pos in self.visited[:-5]:
                if pos != self.start and pos != self.goal and pos != self.current_pos:
                    display[pos] = 'o'
        
        # Marquer positions importantes
        display[self.start] = 'S'
        display[self.goal] = 'G'
        display[self.current_pos] = 'A'
        
        if mode == 'human':
            print('\n' + '=' * (self.width * 2 + 1))
            for row in display:
                print('|' + ' '.join(row) + '|')
            print('=' * (self.width * 2 + 1))
            print(f"Steps: {self.steps}/{self.max_steps} | Position: {self.current_pos}")
        
        return display