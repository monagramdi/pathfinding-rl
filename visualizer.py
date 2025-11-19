import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class MazeVisualizer:
    """Outils de visualisation pour les labyrinthes et agents."""
    
    @staticmethod
    def plot_maze(maze, start=None, goal=None, path=None, title="Maze", 
                  save_path=None, show=True):
        """
        Affiche un labyrinthe avec options.
        
        Args:
            maze: Grille numpy (0=libre, 1=mur)
            start: Position de départ (row, col)
            goal: Position objectif (row, col)
            path: Liste de positions du chemin
            title: Titre du graphique
            save_path: Chemin de sauvegarde
            show: Afficher le graphique
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Créer matrice de couleurs
        display = np.ones((*maze.shape, 3))  # RGB
        
        # Murs en noir
        display[maze == 1] = [0, 0, 0]
        
        # Chemins libres en blanc
        display[maze == 0] = [1, 1, 1]
        
        # Chemin en jaune
        if path:
            for pos in path:
                display[pos] = [1, 1, 0]
        
        # Start en vert
        if start:
            display[start] = [0, 1, 0]
        
        # Goal en rouge
        if goal:
            display[goal] = [1, 0, 0]
        
        ax.imshow(display, interpolation='nearest')
        
        # Grille
        ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Légende
        legend_elements = [
            mpatches.Patch(color='green', label='Start'),
            mpatches.Patch(color='red', label='Goal'),
            mpatches.Patch(color='yellow', label='Path'),
            mpatches.Patch(color='black', label='Wall'),
            mpatches.Patch(color='white', label='Free')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    @staticmethod
    def plot_q_values(agent, maze, start, goal, save_path=None):
        """Visualise les Q-values pour chaque état."""
        height, width = maze.shape
        
        # Créer 4 subplots pour les 4 actions
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        action_names = ['Up', 'Down', 'Left', 'Right']
        
        for action_idx, (ax, name) in enumerate(zip(axes.flat, action_names)):
            q_map = np.zeros((height, width))
            
            for i in range(height):
                for j in range(width):
                    if maze[i, j] == 0:  # Case libre
                        q_map[i, j] = agent.q_table[(i, j)][action_idx]
            
            im = ax.imshow(q_map, cmap='RdYlGn', interpolation='nearest')
            ax.set_title(f'Q-Values - Action: {name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            # Marquer start et goal
            if start:
                ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
            if goal:
                ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal')
            
            ax.legend()
            plt.colorbar(im, ax=ax)
            
            # Grille
            ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Q-values sauvegardées dans {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_value_function(agent, maze, start, goal, save_path=None):
        """Visualise la fonction de valeur (max Q-value par état)."""
        height, width = maze.shape
        value_map = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                if maze[i, j] == 0:  # Case libre
                    value_map[i, j] = agent.get_value((i, j))
                else:
                    value_map[i, j] = np.nan  # Murs
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(value_map, cmap='viridis', interpolation='nearest')
        ax.set_title('Value Function (Max Q-Value per State)', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        
        # Marquer start et goal
        if start:
            ax.plot(start[1], start[0], 'wo', markersize=15, 
                   markeredgecolor='black', markeredgewidth=2, label='Start')
        if goal:
            ax.plot(goal[1], goal[0], 'w*', markersize=20, 
                   markeredgecolor='black', markeredgewidth=2, label='Goal')
        
        ax.legend(fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Value', fontsize=12)
        
        # Grille
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', size=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Value function sauvegardée dans {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_policy(agent, maze, start, goal, save_path=None):
        """Visualise la politique (flèches indiquant la meilleure action)."""
        height, width = maze.shape
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Afficher le labyrinthe
        display = maze.copy().astype(float)
        ax.imshow(display, cmap='binary', interpolation='nearest', alpha=0.3)
        
        # Directions pour les flèches
        arrow_directions = {
            0: (0, -0.3),   # Up
            1: (0, 0.3),    # Down
            2: (-0.3, 0),   # Left
            3: (0.3, 0)     # Right
        }
        
        # Dessiner les flèches
        for i in range(height):
            for j in range(width):
                if maze[i, j] == 0 and (i, j) != goal:  # Case libre, pas le goal
                    q_values = agent.q_table[(i, j)]
                    best_action = np.argmax(q_values)
                    
                    if np.sum(q_values) > 0:  # Si l'état a été visité
                        dx, dy = arrow_directions[best_action]
                        ax.arrow(j, i, dx, dy, head_width=0.2, head_length=0.15,
                                fc='blue', ec='blue', alpha=0.7)
        
        # Marquer start et goal
        if start:
            ax.plot(start[1], start[0], 'go', markersize=20, label='Start')
        if goal:
            ax.plot(goal[1], goal[0], 'r*', markersize=25, label='Goal')
        
        ax.set_title('Policy (Best Action per State)', fontsize=16, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.legend(fontsize=12)
        
        # Grille
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        ax.set_xlim([-0.5, width - 0.5])
        ax.set_ylim([height - 0.5, -0.5])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Policy sauvegardée dans {save_path}")
        
        plt.show()
    
    @staticmethod
    def compare_paths(maze, paths_dict, start, goal, save_path=None):
        """
        Compare plusieurs chemins sur le même labyrinthe.
        
        Args:
            maze: Grille numpy
            paths_dict: Dict {nom: chemin} avec plusieurs chemins à comparer
            start: Position de départ
            goal: Position objectif
            save_path: Chemin de sauvegarde
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Afficher le labyrinthe
        display = maze.copy().astype(float)
        ax.imshow(display, cmap='binary', interpolation='nearest', alpha=0.3)
        
        # Couleurs pour les différents chemins
        colors = plt.cm.rainbow(np.linspace(0, 1, len(paths_dict)))
        
        # Tracer chaque chemin
        for (name, path), color in zip(paths_dict.items(), colors):
            if path:
                path_array = np.array(path)
                ax.plot(path_array[:, 1], path_array[:, 0], 
                       color=color, linewidth=3, alpha=0.7, 
                       label=f'{name} ({len(path)} steps)', marker='o', markersize=4)
        
        # Marquer start et goal
        if start:
            ax.plot(start[1], start[0], 'go', markersize=20, 
                   markeredgecolor='black', markeredgewidth=2, label='Start')
        if goal:
            ax.plot(goal[1], goal[0], 'r*', markersize=25, 
                   markeredgecolor='black', markeredgewidth=2, label='Goal')
        
        ax.set_title('Path Comparison', fontsize=16, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparaison sauvegardée dans {save_path}")
        
        plt.show()