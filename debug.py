"""
Script de diagnostic pour vérifier que tout fonctionne correctement.
"""
import numpy as np
from environment import MazeEnvironment
from maze_generator import MazeGenerator
from agent import QLearningAgent
from utils import astar
from visualizer import MazeVisualizer

def test_environment():
    """Test 1: Vérifier que l'environnement fonctionne."""
    print("\n" + "="*60)
    print("TEST 1: ENVIRONNEMENT")
    print("="*60)
    
    maze = MazeGenerator.empty_maze(5, 5)
    start = (0, 0)
    goal = (4, 4)
    env = MazeEnvironment(maze, start, goal)
    
    print("✓ Environnement créé")
    print(f"  Dimensions: {env.height}x{env.width}")
    print(f"  Actions possibles: {env.n_actions}")
    print(f"  Max steps: {env.max_steps}")
    
    # Test d'un épisode manuel
    state = env.reset()
    total_reward = 0
    
    # Aller vers le goal (bas + droite)
    actions = [1, 3, 1, 3, 1, 3, 1, 3]  # Bas, Droite alternés
    
    for action in actions:
        state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    if info['reason'] == 'goal_reached':
        print("✅ Goal atteint manuellement")
    else:
        print(f"⚠️  Problème: {info['reason']}")
    
    print(f"  Steps: {env.steps}, Reward: {total_reward:.2f}")
    return True


def test_maze_generation():
    """Test 2: Vérifier la génération de labyrinthes."""
    print("\n" + "="*60)
    print("TEST 2: GÉNÉRATION DE LABYRINTHES")
    print("="*60)
    
    types = {
        'Empty': MazeGenerator.empty_maze(10, 10),
        'Simple': MazeGenerator.simple_maze(10, 10, 0.2),
        'Corridor': MazeGenerator.corridor_maze(10, 10),
    }
    
    for name, maze in types.items():
        start = (0, 0)
        goal = (9, 9)
        maze = MazeGenerator.ensure_path_exists(maze, start, goal)
        path = astar(maze, start, goal)
        
        if path:
            print(f"✅ {name:10s}: chemin existe ({len(path)} steps)")
        else:
            print(f"❌ {name:10s}: PAS de chemin!")
    
    return True


def test_untrained_agent():
    """Test 3: Agent non entraîné (devrait échouer)."""
    print("\n" + "="*60)
    print("TEST 3: AGENT NON ENTRAÎNÉ")
    print("="*60)
    
    maze = MazeGenerator.empty_maze(5, 5)
    start = (0, 0)
    goal = (4, 4)
    env = MazeEnvironment(maze, start, goal)
    
    agent = QLearningAgent(n_actions=4)
    
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = agent.get_action(state, training=False)
        state, reward, done, info = env.step(action)
        steps += 1
    
    print(f"Agent non entraîné: {info['reason']} après {steps} steps")
    print("⚠️  C'est normal qu'il échoue (pas encore entraîné)")
    return True


def test_quick_training():
    """Test 4: Entraînement rapide sur labyrinthe simple."""
    print("\n" + "="*60)
    print("TEST 4: ENTRAÎNEMENT RAPIDE (100 épisodes)")
    print("="*60)
    
    # Labyrinthe très simple
    maze = MazeGenerator.empty_maze(5, 5)
    start = (0, 0)
    goal = (4, 4)
    env = MazeEnvironment(maze, start, goal)
    
    # Agent
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        epsilon_min=0.1
    )
    
    # Entraînement rapide
    successes = 0
    for episode in range(100):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
        
        if info['reason'] == 'goal_reached':
            successes += 1
        
        agent.decay_epsilon()
    
    success_rate = (successes / 100) * 100
    print(f"Taux de succès: {success_rate:.1f}%")
    
    # Test de l'agent entraîné
    print("\nTest de l'agent entraîné:")
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 100:
        action = agent.get_action(state, training=False)
        state, reward, done, info = env.step(action)
        steps += 1
    
    if info['reason'] == 'goal_reached':
        print(f"✅ Agent entraîné réussit en {steps} steps!")
        return True
    else:
        print(f"⚠️  Agent entraîné a échoué: {info['reason']}")
        if success_rate < 50:
            print("❌ PROBLÈME: Le taux de succès est trop faible")
            return False
        return True


def test_astar_baseline():
    """Test 5: Vérifier que A* fonctionne."""
    print("\n" + "="*60)
    print("TEST 5: A* BASELINE")
    print("="*60)
    
    maze = MazeGenerator.simple_maze(10, 10, 0.2)
    start = (0, 0)
    goal = (9, 9)
    maze = MazeGenerator.ensure_path_exists(maze, start, goal)
    
    path = astar(maze, start, goal)
    
    if path:
        print(f"✅ A* trouve un chemin en {len(path)} steps")
        return True
    else:
        print("❌ PROBLÈME: A* ne trouve pas de chemin")
        return False


def test_saved_agent():
    """Test 6: Vérifier l'agent sauvegardé."""
    print("\n" + "="*60)
    print("TEST 6: AGENT SAUVEGARDÉ")
    print("="*60)
    
    try:
        agent = QLearningAgent(n_actions=4)
        agent.load('saved_models/qlearning_agent.pkl')
        
        stats = agent.get_stats()
        print(f"✅ Agent chargé avec succès")
        print(f"  États explorés: {stats['n_states']}")
        print(f"  Steps d'entraînement: {stats['training_steps']}")
        print(f"  Epsilon actuel: {stats['epsilon']:.3f}")
        
        if stats['n_states'] < 50:
            print("⚠️  WARNING: Très peu d'états explorés!")
            print("   → L'agent n'a probablement pas assez appris")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ Aucun agent sauvegardé trouvé")
        print("   → Lancer 'python train.py' d'abord")
        return False


def test_full_pipeline():
    """Test 7: Pipeline complet sur un labyrinthe simple."""
    print("\n" + "="*60)
    print("TEST 7: PIPELINE COMPLET")
    print("="*60)
    
    # Utiliser le MÊME labyrinthe que pour l'entraînement
    maze = MazeGenerator.simple_maze(10, 10, obstacle_ratio=0.2)
    start = (0, 0)
    goal = (9, 9)
    maze = MazeGenerator.ensure_path_exists(maze, start, goal)
    
    # Vérifier qu'un chemin existe
    optimal_path = astar(maze, start, goal)
    print(f"Chemin optimal: {len(optimal_path)} steps")
    
    # Tester avec agent sauvegardé
    try:
        agent = QLearningAgent(n_actions=4)
        agent.load('saved_models/qlearning_agent.pkl')
        
        env = MazeEnvironment(maze, start, goal)
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 500:
            action = agent.get_action(state, training=False)
            state, reward, done, info = env.step(action)
            steps += 1
        
        if info['reason'] == 'goal_reached':
            print(f"✅ SUCCÈS: Agent trouve le goal en {steps} steps")
            efficiency = (len(optimal_path) / steps) * 100
            print(f"   Efficacité: {efficiency:.1f}% de l'optimal")
            return True
        else:
            print(f"⚠️  ÉCHEC: {info['reason']} après {steps} steps")
            print(f"   (Chemin optimal = {len(optimal_path)} steps)")
            return False
            
    except FileNotFoundError:
        print("⚠️  Pas d'agent sauvegardé - skip ce test")
        return True


def print_recommendations(results):
    """Affiche des recommandations basées sur les résultats."""
    print("\n" + "="*60)
    print("RECOMMANDATIONS")
    print("="*60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ Tous les tests passent!")
        print("\nTon projet est prêt à être livré si:")
        print("  1. Le taux de succès final > 80% (voir train.py)")
        print("  2. L'agent trouve le goal sur des labyrinthes simples")
        print("  3. Les visualisations sont claires")
    else:
        print("⚠️  Certains tests échouent. Voici quoi faire:\n")
        
        if not results.get('test_saved_agent', True):
            print("❌ PRIORITÉ 1: Entraîner un agent")
            print("   → python train.py")
            print("   → Attendre 2000 épisodes minimum")
            print("   → Vérifier que le taux de succès final > 70%\n")
        
        if not results.get('test_quick_training', True):
            print("❌ PRIORITÉ 2: Problème d'apprentissage")
            print("   → Vérifier les hyperparamètres dans train.py")
            print("   → Alpha = 0.1, Gamma = 0.95")
            print("   → Epsilon decay = 0.995\n")
        
        if not results.get('test_full_pipeline', True):
            print("⚠️  L'agent échoue sur des labyrinthes standards")
            print("   → Augmenter le nombre d'épisodes (5000+)")
            print("   → Ou commencer avec des labyrinthes plus simples")
            print("   → Vérifier le reward shaping dans environment.py\n")
    
    print("\n" + "="*60)
    print("CHECKLIST AVANT LIVRAISON")
    print("="*60)
    print("[ ] Code propre et commenté")
    print("[ ] README.md à jour avec instructions")
    print("[ ] Agent entraîné et sauvegardé")
    print("[ ] Taux de succès > 70% sur labyrinthe simple")
    print("[ ] Graphiques d'entraînement générés")
    print("[ ] Comparaison avec A* effectuée")
    print("[ ] Visualisations fonctionnelles")
    print("[ ] Requirements.txt à jour")
    print("[ ] Tests passent (python debug.py)")


def main():
    """Lance tous les tests de diagnostic."""
    print("="*60)
    print("DIAGNOSTIC COMPLET DU PROJET PATHFINDING")
    print("="*60)
    
    tests = [
        ('test_environment', test_environment),
        ('test_maze_generation', test_maze_generation),
        ('test_untrained_agent', test_untrained_agent),
        ('test_quick_training', test_quick_training),
        ('test_astar_baseline', test_astar_baseline),
        ('test_saved_agent', test_saved_agent),
        ('test_full_pipeline', test_full_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n❌ ERREUR dans {test_name}: {str(e)}")
            results[test_name] = False
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nScore: {passed}/{total} tests réussis")
    
    # Recommandations
    print_recommendations(results)


if __name__ == "__main__":
    main()