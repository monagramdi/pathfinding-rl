import numpy as np
import matplotlib.pyplot as plt
from environment import MazeEnvironment
from maze_generator import MazeGenerator
from agent import QLearningAgent
import time

def test_agent(env: MazeEnvironment, agent: QLearningAgent, 
               render: bool = True, delay: float = 0.2, max_steps: int = 200,
               test_epsilon: float = 0.1) -> dict:
    """
    Teste l'agent sur un épisode.
    
    Args:
        env: Environnement
        agent: Agent entraîné
        render: Afficher l'environnement
        delay: Délai entre les steps (secondes)
        max_steps: Nombre maximum de steps (sécurité)
        test_epsilon: Epsilon pour le test (0.1 = 10% exploration)
    
    Returns:
        results: Dictionnaire avec résultats du test
    """
    state = env.reset()
    done = False
    total_reward = 0
    path = [state]
    steps = 0
    
    # Sauvegarder l'epsilon original et le remplacer temporairement
    original_epsilon = agent.epsilon
    agent.epsilon = test_epsilon  # Ajouter un peu d'exploration pour le test
    
    if render:
        print("\n" + "=" * 60)
        print("TEST - Exécution de l'agent entraîné")
        print(f"(avec epsilon={test_epsilon} pour éviter les blocages)")
        print("=" * 60)
        env.render()
        time.sleep(delay)
    
    while not done and steps < max_steps:
        # Action avec un peu d'exploration
        action = agent.get_action(state, training=True)  # training=True pour utiliser epsilon
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        path.append(next_state)
        state = next_state
        steps += 1
        
        if render:
            env.render()
            action_names = ['Haut', 'Bas', 'Gauche', 'Droite']
            print(f"Step {steps}/{max_steps} | Action: {action_names[action]} | Reward: {reward:.2f}")
            time.sleep(delay)
    
    # Restaurer l'epsilon original
    agent.epsilon = original_epsilon
    
    # Vérifier si timeout
    if steps >= max_steps and not done:
        info = {'reason': 'timeout'}
        if render:
            print("\n⏱️  TIMEOUT - Limite de steps atteinte!")
    
    if render:
        print("\n" + "=" * 60)
        print("RÉSULTATS")
        print("=" * 60)
        print(f"Raison d'arrêt: {info['reason']}")
        print(f"Steps total: {steps}")
        print(f"Récompense totale: {total_reward:.2f}")
        success_reasons = ['goal_reached']
        print(f"Succès: {'✅ OUI' if info['reason'] in success_reasons else '❌ NON'}")
    
    results = {
        'success': info['reason'] == 'goal_reached',
        'steps': steps,
        'reward': total_reward,
        'path': path,
        'reason': info['reason']
    }
    
    return results


def visualize_path(env: MazeEnvironment, path: list, save_path: str = None):
    """Visualise le chemin trouvé par l'agent."""
    maze_display = env.maze.copy().astype(float)
    
    # Colorer le chemin
    for i, (row, col) in enumerate(path):
        intensity = i / len(path)  # Gradient du début à la fin
        maze_display[row, col] = -intensity
    
    # Marquer start et goal
    maze_display[env.start] = -2
    maze_display[env.goal] = -3
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Afficher le labyrinthe
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(maze_display, cmap=cmap, interpolation='nearest')
    
    # Ajouter grille
    ax.set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    # Labels
    ax.set_title(f'Chemin trouvé par l\'agent ({len(path)} steps)', fontsize=16, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Start'),
        Patch(facecolor='red', label='Goal'),
        Patch(facecolor='yellow', label='Chemin'),
        Patch(facecolor='black', label='Mur')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Ajouter numéros sur le chemin (optionnel, pour petits labyrinthes)
    if len(path) < 50:
        for i, (row, col) in enumerate(path[::max(1, len(path)//20)]):
            ax.text(col, row, str(i), ha='center', va='center', 
                   color='white', fontsize=8, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Progression')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualisation sauvegardée dans {save_path}")
    
    plt.show()


def evaluate_agent(env: MazeEnvironment, agent: QLearningAgent, 
                   n_tests: int = 100, test_epsilon: float = 0.1) -> dict:
    """
    Évalue l'agent sur plusieurs épisodes.
    
    Args:
        test_epsilon: Epsilon pour l'évaluation (petit pour éviter blocages)
    
    Returns:
        stats: Statistiques d'évaluation
    """
    print(f"\n{'='*60}")
    print(f"ÉVALUATION SUR {n_tests} ÉPISODES")
    print(f"{'='*60}")
    
    successes = 0
    total_steps = []
    total_rewards = []
    
    for i in range(n_tests):
        results = test_agent(env, agent, render=False, test_epsilon=test_epsilon)
        
        if results['success']:
            successes += 1
            total_steps.append(results['steps'])
        
        total_rewards.append(results['reward'])
        
        if (i + 1) % 20 == 0:
            print(f"Progression: {i+1}/{n_tests} épisodes")
    
    success_rate = (successes / n_tests) * 100
    avg_steps = np.mean(total_steps) if total_steps else 0
    avg_reward = np.mean(total_rewards)
    
    print(f"\n{'='*60}")
    print("RÉSULTATS D'ÉVALUATION")
    print(f"{'='*60}")
    print(f"Taux de succès: {success_rate:.1f}% ({successes}/{n_tests})")
    print(f"Steps moyens (succès): {avg_steps:.1f}")
    print(f"Récompense moyenne: {avg_reward:.2f}")
    print(f"Meilleur: {min(total_steps) if total_steps else 'N/A'} steps")
    print(f"Pire: {max(total_steps) if total_steps else 'N/A'} steps")
    
    stats = {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'all_steps': total_steps,
        'all_rewards': total_rewards
    }
    
    return stats


def main():
    """Fonction principale de test."""
    print("=" * 60)
    print("PATHFINDING Q-LEARNING - TESTING")
    print("=" * 60)
    
    # Configuration
    HEIGHT, WIDTH = 10, 10
    
    # Charger le labyrinthe utilisé pendant l'entraînement
    print("\n[1/3] Chargement du labyrinthe d'entraînement...")
    try:
        maze_data = np.load('saved_models/training_maze.npz')
        maze = maze_data['maze']
        start = tuple(maze_data['start'])
        goal = tuple(maze_data['goal'])
        print(f"  ✅ Labyrinthe d'entraînement chargé")
    except FileNotFoundError:
        print("  ⚠️  Pas de labyrinthe sauvegardé, génération aléatoire...")
        # IMPORTANT : Utiliser le MÊME type que dans train.py !
        # maze = MazeGenerator.empty_maze(HEIGHT, WIDTH)
        # ou 
        maze = MazeGenerator.simple_maze(HEIGHT, WIDTH, obstacle_ratio=0.2)
        start = (0, 0)
        goal = (HEIGHT - 1, WIDTH - 1)
        maze = MazeGenerator.ensure_path_exists(maze, start, goal)
    
    env = MazeEnvironment(maze, start, goal)
    print("  Labyrinthe chargé!")
    
    # Charger l'agent
    print("\n[2/3] Chargement de l'agent entraîné...")
    agent = QLearningAgent(n_actions=4)
    try:
        agent.load('saved_models/qlearning_agent.pkl')
        print("  Agent chargé avec succès!")
        agent_stats = agent.get_stats()
        print(f"    États explorés: {agent_stats['n_states']}")
        print(f"    Training steps: {agent_stats['training_steps']}")
    except FileNotFoundError:
        print("  ⚠️  Aucun agent sauvegardé trouvé!")
        print("  Veuillez d'abord entraîner un agent avec train.py")
        return
    
    # Test visuel
    print("\n[3/3] Test de l'agent...")
    results = test_agent(env, agent, render=True, delay=0.3, max_steps=200, test_epsilon=0.1)
    
    # Visualiser le chemin
    if results['success']:
        print("\n[4/4] Visualisation du chemin...")
        visualize_path(env, results['path'], save_path='results/path_visualization.png')
    
    # Évaluation sur plusieurs épisodes
    print("\n[5/5] Évaluation approfondie...")
    eval_stats = evaluate_agent(env, agent, n_tests=100)
    
    print("\n✅ Test terminé avec succès!")


if __name__ == "__main__":
    main()