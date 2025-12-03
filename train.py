import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from environment import MazeEnvironment
from maze_generator import MazeGenerator
from agent import QLearningAgent

def train_agent(env: MazeEnvironment, agent: QLearningAgent, 
                n_episodes: int = 1000, verbose: bool = True):
    """
    Entraîne l'agent sur l'environnement.
    
    Args:
        env: Environnement de labyrinthe
        agent: Agent Q-Learning
        n_episodes: Nombre d'épisodes d'entraînement
        verbose: Afficher les logs
    
    Returns:
        stats: Dictionnaire avec statistiques d'entraînement
    """
    rewards_history = []
    steps_history = []
    success_history = []
    epsilon_history = []
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Choisir et exécuter action
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Mettre à jour l'agent
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
        
        # Décrémenter epsilon
        agent.decay_epsilon()
        
        # Sauvegarder statistiques
        rewards_history.append(episode_reward)
        steps_history.append(env.steps)
        success_history.append(1 if info['reason'] == 'goal_reached' else 0)
        epsilon_history.append(agent.epsilon)
        
        # Log périodique
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            success_rate = np.mean(success_history[-100:]) * 100
            avg_steps = np.mean(steps_history[-100:])
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
    
    stats = {
        'rewards': rewards_history,
        'steps': steps_history,
        'success': success_history,
        'epsilon': epsilon_history
    }
    
    return stats


def plot_training_stats(stats: dict, save_path: str = None):
    """Affiche les statistiques d'entraînement."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Récompenses
    axes[0, 0].plot(stats['rewards'], alpha=0.3, label='Raw')
    window = min(100, len(stats['rewards']) // 10)
    if window > 0:
        smoothed = np.convolve(stats['rewards'], np.ones(window)/window, mode='valid')
        axes[0, 0].plot(smoothed, label=f'Moving Avg ({window})', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Rewards per Episode')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps
    axes[0, 1].plot(stats['steps'], alpha=0.3, label='Raw')
    if window > 0:
        smoothed_steps = np.convolve(stats['steps'], np.ones(window)/window, mode='valid')
        axes[0, 1].plot(smoothed_steps, label=f'Moving Avg ({window})', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Taux de succès (fenêtre glissante)
    success_rate = []
    window_size = min(50, len(stats['success']) // 10)
    for i in range(len(stats['success']) - window_size):
        rate = np.mean(stats['success'][i:i+window_size]) * 100
        success_rate.append(rate)
    axes[1, 0].plot(success_rate, linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Success Rate (%)')
    axes[1, 0].set_title(f'Success Rate (window={window_size})')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 105])
    
    # Epsilon
    axes[1, 1].plot(stats['epsilon'], linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Graphiques sauvegardés dans {save_path}")
    
    plt.show()


def main():
    """Fonction principale d'entraînement."""
    print("=" * 60)
    print("PATHFINDING Q-LEARNING - TRAINING")
    print("=" * 60)
    
    # Configuration
    HEIGHT, WIDTH = 10, 10
    N_EPISODES = 10000
    
    # Créer le labyrinthe
    print("\n[1/4] Génération du labyrinthe...")
    maze = MazeGenerator.simple_maze(HEIGHT, WIDTH, obstacle_ratio=0.2)
    
    # Positions start et goal
    start = (0, 0)
    goal = (HEIGHT - 1, WIDTH - 1)
    
    # S'assurer qu'un chemin existe
    maze = MazeGenerator.ensure_path_exists(maze, start, goal)
    
    print(f"  Labyrinthe: {HEIGHT}x{WIDTH}")
    print(f"  Start: {start}, Goal: {goal}")
    
    # Créer environnement
    print("\n[2/4] Création de l'environnement...")
    env = MazeEnvironment(maze, start, goal)
    print("  Environnement créé!")
    
    # Afficher le labyrinthe
    print("\nLabyrinthe:")
    env.render()
    
    # Créer agent
    print("\n[3/4] Initialisation de l'agent...")
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.998,
        epsilon_min=0.01
    )
    print("  Agent Q-Learning initialisé!")
    
    # Entraînement
    print(f"\n[4/4] Entraînement sur {N_EPISODES} épisodes...")
    stats = train_agent(env, agent, n_episodes=N_EPISODES, verbose=True)
    
    # Résumé final
    print("\n" + "=" * 60)
    print("TRAINING TERMINÉ!")
    print("=" * 60)
    final_success = np.mean(stats['success'][-100:]) * 100
    final_avg_reward = np.mean(stats['rewards'][-100:])
    final_avg_steps = np.mean(stats['steps'][-100:])
    
    print(f"\nPerformance finale (100 derniers épisodes):")
    print(f"  Taux de succès: {final_success:.1f}%")
    print(f"  Récompense moyenne: {final_avg_reward:.2f}")
    print(f"  Steps moyens: {final_avg_steps:.1f}")
    print(f"  États explorés: {len(agent.q_table)}")
    
    # Sauvegarder l'agent
    print("\n[5/5] Sauvegarde de l'agent...")
    agent.save('saved_models/qlearning_agent.pkl')
    
    # Afficher graphiques
    print("\nAffichage des statistiques...")
    plot_training_stats(stats, save_path='results/training_stats.png')
    
    print("\n✅ Entraînement terminé avec succès!")
    print("Utilisez test.py pour évaluer l'agent.")


if __name__ == "__main__":
    main()