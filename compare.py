import numpy as np
import time
from environment import MazeEnvironment
from maze_generator import MazeGenerator
from agent import QLearningAgent
from utils import astar, bfs, dijkstra, calculate_path_metrics, evaluate_solution_quality
from visualizer import MazeVisualizer

def get_rl_path(env: MazeEnvironment, agent: QLearningAgent) -> list:
    """Obtient le chemin suivi par l'agent RL."""
    state = env.reset()
    path = [state]
    done = False
    max_steps = 500  # Sécurité
    
    steps = 0
    while not done and steps < max_steps:
        action = agent.get_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        path.append(next_state)
        state = next_state
        steps += 1
    
    if info.get('reason') == 'goal_reached':
        return path
    else:
        return []  # Échec


def compare_algorithms(maze: np.ndarray, start: tuple, goal: tuple, 
                       agent: QLearningAgent = None) -> dict:
    """
    Compare l'agent RL avec les algorithmes classiques.
    
    Returns:
        results: Dict avec chemins et statistiques
    """
    print("\n" + "=" * 60)
    print("COMPARAISON DES ALGORITHMES")
    print("=" * 60)
    
    results = {}
    
    # A*
    print("\n[1/4] Exécution de A*...")
    start_time = time.time()
    astar_path = astar(maze, start, goal)
    astar_time = time.time() - start_time
    
    if astar_path:
        astar_metrics = calculate_path_metrics(astar_path, maze)
        print(f"  ✅ A* trouvé un chemin en {len(astar_path)} steps")
        print(f"     Temps: {astar_time*1000:.2f} ms")
        print(f"     Efficacité: {astar_metrics['efficiency']:.2%}")
    else:
        print("  ❌ A* n'a pas trouvé de chemin")
        astar_metrics = {}
    
    results['A*'] = {
        'path': astar_path,
        'time': astar_time,
        'metrics': astar_metrics
    }
    
    # BFS
    print("\n[2/4] Exécution de BFS...")
    start_time = time.time()
    bfs_path = bfs(maze, start, goal)
    bfs_time = time.time() - start_time
    
    if bfs_path:
        bfs_metrics = calculate_path_metrics(bfs_path, maze)
        print(f"  ✅ BFS trouvé un chemin en {len(bfs_path)} steps")
        print(f"     Temps: {bfs_time*1000:.2f} ms")
        print(f"     Efficacité: {bfs_metrics['efficiency']:.2%}")
    else:
        print("  ❌ BFS n'a pas trouvé de chemin")
        bfs_metrics = {}
    
    results['BFS'] = {
        'path': bfs_path,
        'time': bfs_time,
        'metrics': bfs_metrics
    }
    
    # Dijkstra
    print("\n[3/4] Exécution de Dijkstra...")
    start_time = time.time()
    dijkstra_path = dijkstra(maze, start, goal)
    dijkstra_time = time.time() - start_time
    
    if dijkstra_path:
        dijkstra_metrics = calculate_path_metrics(dijkstra_path, maze)
        print(f"  ✅ Dijkstra trouvé un chemin en {len(dijkstra_path)} steps")
        print(f"     Temps: {dijkstra_time*1000:.2f} ms")
        print(f"     Efficacité: {dijkstra_metrics['efficiency']:.2%}")
    else:
        print("  ❌ Dijkstra n'a pas trouvé de chemin")
        dijkstra_metrics = {}
    
    results['Dijkstra'] = {
        'path': dijkstra_path,
        'time': dijkstra_time,
        'metrics': dijkstra_metrics
    }
    
    # RL Agent
    if agent:
        print("\n[4/4] Exécution de l'agent RL...")
        env = MazeEnvironment(maze, start, goal)
        start_time = time.time()
        rl_path = get_rl_path(env, agent)
        rl_time = time.time() - start_time
        
        if rl_path:
            rl_metrics = calculate_path_metrics(rl_path, maze)
            print(f"  ✅ RL Agent trouvé un chemin en {len(rl_path)} steps")
            print(f"     Temps: {rl_time*1000:.2f} ms")
            print(f"     Efficacité: {rl_metrics['efficiency']:.2%}")
            
            # Comparer avec l'optimal (A*)
            if astar_path:
                quality = evaluate_solution_quality(rl_path, astar_path)
                print(f"     Qualité vs optimal: {quality['percentage_optimal']:.1f}%")
                print(f"     Steps supplémentaires: {quality['extra_steps']}")
        else:
            print("  ❌ RL Agent n'a pas trouvé de chemin")
            rl_metrics = {}
        
        results['RL Agent'] = {
            'path': rl_path,
            'time': rl_time,
            'metrics': rl_metrics
        }
    
    return results


def print_summary_table(results: dict):
    """Affiche un tableau récapitulatif."""
    print("\n" + "=" * 90)
    print("TABLEAU RÉCAPITULATIF")
    print("=" * 90)
    print(f"{'Algorithme':<15} {'Steps':<8} {'Efficacité':<12} {'Virages':<10} "
          f"{'Temps (ms)':<12} {'Succès':<8}")
    print("-" * 90)
    
    for name, data in results.items():
        path = data.get('path', [])
        time_ms = data.get('time', 0) * 1000
        metrics = data.get('metrics', {})
        success = "✅" if path else "❌"
        
        if path:
            print(f"{name:<15} {metrics.get('length', 0):<8} "
                  f"{metrics.get('efficiency', 0):<12.2%} "
                  f"{metrics.get('turns', 0):<10} "
                  f"{time_ms:<12.3f} {success:<8}")
        else:
            print(f"{name:<15} {'N/A':<8} {'N/A':<12} {'N/A':<10} "
                  f"{time_ms:<12.3f} {success:<8}")
    
    print("=" * 90)


def analyze_performance(results: dict):
    """Analyse détaillée des performances."""
    print("\n" + "=" * 60)
    print("ANALYSE DES PERFORMANCES")
    print("=" * 60)
    
    # Trouver le meilleur
    best_length = min([len(r['path']) for r in results.values() if r['path']], 
                     default=float('inf'))
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    
    print(f"\n🏆 Chemin le plus court: {best_length} steps")
    algos_optimal = [name for name, data in results.items() 
                    if data['path'] and len(data['path']) == best_length]
    print(f"   Algorithme(s): {', '.join(algos_optimal)}")
    
    print(f"\n⚡ Algorithme le plus rapide: {fastest[0]}")
    print(f"   Temps: {fastest[1]['time']*1000:.3f} ms")
    
    # Analyse de la qualité RL
    if 'RL Agent' in results and 'A*' in results:
        rl_path = results['RL Agent']['path']
        astar_path = results['A*']['path']
        
        if rl_path and astar_path:
            quality = evaluate_solution_quality(rl_path, astar_path)
            print(f"\n🤖 Performance RL vs Optimal:")
            print(f"   Qualité: {quality['percentage_optimal']:.1f}%")
            print(f"   Steps RL: {quality['rl_length']}")
            print(f"   Steps optimal: {quality['optimal_length']}")
            print(f"   Différence: +{quality['extra_steps']} steps")
            
            if quality['percentage_optimal'] >= 100:
                print("   🎉 L'agent RL a trouvé le chemin optimal!")
            elif quality['percentage_optimal'] >= 90:
                print("   ✅ L'agent RL est très proche de l'optimal")
            elif quality['percentage_optimal'] >= 70:
                print("   ⚠️  L'agent RL peut être amélioré")
            else:
                print("   ❌ L'agent RL nécessite plus d'entraînement")


def main():
    """Fonction principale de comparaison."""
    print("=" * 60)
    print("COMPARAISON : RL vs ALGORITHMES CLASSIQUES")
    print("=" * 60)
    
    # Configuration
    HEIGHT, WIDTH = 10, 10
    
    # Créer le labyrinthe
    print("\n[Setup] Génération du labyrinthe...")
    maze = MazeGenerator.simple_maze(HEIGHT, WIDTH, obstacle_ratio=0.2)
    start = (0, 0)
    goal = (HEIGHT - 1, WIDTH - 1)
    maze = MazeGenerator.ensure_path_exists(maze, start, goal)
    print(f"  Labyrinthe: {HEIGHT}x{WIDTH}")
    
    # Charger l'agent RL
    print("\n[Setup] Chargement de l'agent RL...")
    agent = QLearningAgent(n_actions=4)
    try:
        agent.load('saved_models/qlearning_agent.pkl')
        print("  ✅ Agent chargé")
    except FileNotFoundError:
        print("  ⚠️  Pas d'agent sauvegardé, comparaison sans RL")
        agent = None
    
    # Comparer les algorithmes
    results = compare_algorithms(maze, start, goal, agent)
    
    # Afficher résultats
    print_summary_table(results)
    analyze_performance(results)
    
    # Visualiser
    print("\n[Visualisation] Comparaison des chemins...")
    paths_dict = {name: data['path'] for name, data in results.items() 
                 if data['path']}
    
    if paths_dict:
        MazeVisualizer.compare_paths(
            maze, paths_dict, start, goal,
            save_path='results/algorithms_comparison.png'
        )
    
    print("\n✅ Comparaison terminée!")


if __name__ == "__main__":
    main()