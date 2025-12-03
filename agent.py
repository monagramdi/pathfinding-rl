import numpy as np
import pickle
from typing import Tuple, Optional
from collections import defaultdict

class QLearningAgent:
    """Agent Q-Learning pour pathfinding."""
    
    def __init__(self, n_actions: int, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Args:
            n_actions: Nombre d'actions possibles
            learning_rate: Taux d'apprentissage (alpha)
            discount_factor: Facteur de discount (gamma)
            epsilon: Taux d'exploration initial
            epsilon_decay: Décroissance de epsilon
            epsilon_min: Epsilon minimal
        """
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: defaultdict pour initialiser automatiquement à 0
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Statistiques
        self.training_steps = 0
        
    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Sélectionne une action selon la politique epsilon-greedy.
        
        Args:
            state: État actuel
            training: Si True, utilise epsilon-greedy, sinon greedy pur
        """
        if training and np.random.random() < self.epsilon:
            # Exploration: action aléatoire
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: meilleure action
            q_values = self.q_table[state]
            
            # Si toutes les Q-values sont égales (état jamais vu), choisir aléatoire
            if np.all(q_values == q_values[0]):
                return np.random.randint(self.n_actions)
            
            # Sinon, prendre la meilleure action
            # Avec tie-breaking aléatoire si plusieurs actions ont le même Q-value max
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state: Tuple[int, int], action: int, reward: float,
               next_state: Tuple[int, int], done: bool):
        """
        Met à jour la Q-table selon l'équation de Bellman.
        
        Q(s,a) = Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            # Si terminal, pas de valeur future
            target_q = reward
        else:
            # Valeur future = récompense + max Q du prochain état
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Mise à jour
        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
        
        self.training_steps += 1
    
    def decay_epsilon(self):
        """Réduit epsilon pour diminuer l'exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Sauvegarde l'agent."""
        data = {
            'q_table': dict(self.q_table),
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent sauvegardé dans {filepath}")
    
    def load(self, filepath: str):
        """Charge un agent sauvegardé."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.n_actions = data['n_actions']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        print(f"Agent chargé depuis {filepath}")
    
    def get_policy(self, state: Tuple[int, int]) -> np.ndarray:
        """Retourne la politique (distribution de probabilités) pour un état."""
        q_values = self.q_table[state]
        if np.sum(q_values) == 0:
            return np.ones(self.n_actions) / self.n_actions
        return q_values / np.sum(q_values)
    
    def get_value(self, state: Tuple[int, int]) -> float:
        """Retourne la valeur d'un état (max Q-value)."""
        return np.max(self.q_table[state])
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur l'agent."""
        return {
            'n_states': len(self.q_table),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }


class SARSAAgent(QLearningAgent):
    """Agent SARSA (on-policy alternative à Q-Learning)."""
    
    def update(self, state: Tuple[int, int], action: int, reward: float,
               next_state: Tuple[int, int], next_action: int, done: bool):
        """
        Mise à jour SARSA: utilise l'action réellement prise (next_action)
        au lieu du max Q comme dans Q-Learning.
        """
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            # SARSA utilise Q(s', a') où a' est l'action choisie
            next_q = self.q_table[next_state][next_action]
            target_q = reward + self.gamma * next_q
        
        self.q_table[state][action] = current_q + self.alpha * (target_q - current_q)
        self.training_steps += 1