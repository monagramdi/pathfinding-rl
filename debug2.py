"""
Script pour debugger le comportement de l'agent en détail.
"""
import numpy as np
from environment import MazeEnvironment
from maze_generator import MazeGenerator
from agent import QLearningAgent

def debug_agent_decision(agent, state):
    """Affiche les Q-values et la décision de l'agent."""
    q_values = agent.q_table[state]
    action = agent.get_action(state, training=False)
    
    print(f"\n  État: {state}")
    print(f"  Q-values: {q_values}")
    print(f"  Max Q: {np.max(q_values):.3f}")
    print(f"  Action choisie: {action} ({['Haut', 'Bas', 'Gauche', 'Droite'][action]})")
    
    # Vérifier si toutes les Q-values sont égales
    if np.all(q_values == q_values[0]):
        print(f"  ⚠️  ÉTAT JAMAIS VU (toutes Q-values = {q_values[0]:.3f})")
    
    return action


def test_with_detailed_debug():
    """Test avec debug détaillé."""
    print("="*60)
    print("DEBUG DÉTAILLÉ DE L'AGENT")
    print("="*60)
    
    # Charger l'agent
    print("\n[1] Chargement de l'agent...")
    agent = QLearningAgent(n_actions=4)
    try:
        agent.load('saved_models/qlearning_agent.pkl')
        stats = agent.get_stats()
        print(f"✅ Agent chargé")
        print(f"   États dans Q-table: {stats['n_states']}")
        print(f"   Epsilon: {stats['epsilon']}")
    except FileNotFoundError:
        print("❌ Pas d'agent sauvegardé!")
        return
    
    # Créer environnement
    print("\n[2] Création du labyrinthe...")
    HEIGHT, WIDTH = 10, 10
    maze = MazeGenerator.simple_maze(HEIGHT, WIDTH, obstacle_ratio=0.2)
    start = (0, 0)
    goal = (HEIGHT - 1, WIDTH - 1)
    maze = MazeGenerator.ensure_path_exists(maze, start, goal)
    
    env = MazeEnvironment(maze, start, goal)
    print(f"✅ Labyrinthe {HEIGHT}x{WIDTH} créé")
    
    # Afficher le labyrinthe
    print("\n[3] Labyrinthe:")
    env.render()
    
    # Test avec debug
    print("\n[4] Test de l'agent (max 30 steps pour debug):")
    print("="*60)
    
    state = env.reset()
    done = False
    steps = 0
    max_steps = 30
    
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    collision_count = 0
    last_states = []
    
    while not done and steps < max_steps:
        steps += 1
        print(f"\n--- STEP {steps} ---")
        
        # Décision de l'agent avec détails
        action = debug_agent_decision(agent, state)
        action_counts[action] += 1
        
        # Exécuter l'action
        next_state, reward, done, info = env.step(action)
        
        print(f"  Résultat:")
        print(f"    Nouvel état: {next_state}")
        print(f"    Reward: {reward:.2f}")
        print(f"    Done: {done}")
        print(f"    Info: {info['reason']}")
        
        # Détecter les patterns problématiques
        if info['reason'] == 'collision':
            collision_count += 1
            print(f"  🧱 COLLISION #{collision_count}")
        
        if next_state == state:
            print(f"  ⚠️  POSITION INCHANGÉE (collision ou bloqué)")
        
        # Tracker les dernières positions
        last_states.append(state)
        if len(last_states) > 5:
            last_states.pop(0)
            # Détecter boucle
            if len(set(last_states)) <= 2:
                print(f"  🔄 BOUCLE DÉTECTÉE: {set(last_states)}")
        
        state = next_state
        
        # Afficher le labyrinthe tous les 5 steps
        if steps % 5 == 0:
            env.render()
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DU DEBUG")
    print("="*60)
    print(f"Steps effectués: {steps}")
    print(f"Raison d'arrêt: {info.get('reason', 'max_steps')}")
    print(f"Position finale: {state}")
    print(f"Goal: {goal}")
    print(f"\nDistribution des actions:")
    for action, count in action_counts.items():
        pct = (count / steps * 100) if steps > 0 else 0
        action_name = ['Haut', 'Bas', 'Gauche', 'Droite'][action]
        print(f"  {action_name}: {count} fois ({pct:.1f}%)")
    print(f"\nCollisions totales: {collision_count}")
    
    # Diagnostic
    print("\n" + "="*60)
    print("DIAGNOSTIC")
    print("="*60)
    
    # Problème 1: Une action dominante
    max_action_count = max(action_counts.values())
    if max_action_count > steps * 0.7:
        dominant_action = max(action_counts, key=action_counts.get)
        print(f"❌ PROBLÈME: Action {dominant_action} ({['Haut', 'Bas', 'Gauche', 'Droite'][dominant_action]}) utilisée {max_action_count}/{steps} fois")
        print(f"   → L'agent répète la même action en boucle")
        print(f"   → Solutions possibles:")
        print(f"      1. Augmenter epsilon pendant le test (test_epsilon=0.2)")
        print(f"      2. Réentraîner avec plus d'exploration")
        print(f"      3. Vérifier que les Q-values sont bien apprises")
    
    # Problème 2: Trop de collisions
    if collision_count > steps * 0.5:
        print(f"❌ PROBLÈME: Trop de collisions ({collision_count}/{steps})")
        print(f"   → L'agent n'a pas appris à éviter les obstacles")
        print(f"   → Solutions:")
        print(f"      1. Réentraîner avec plus d'épisodes (5000+)")
        print(f"      2. Augmenter la pénalité de collision dans environment.py")
    
    # Problème 3: États jamais vus
    print(f"\n📊 Vérification de la couverture de la Q-table:")
    states_checked = 0
    states_learned = 0
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if maze[i, j] == 0:  # Case libre
                states_checked += 1
                q_vals = agent.q_table[(i, j)]
                if not np.all(q_vals == 0):
                    states_learned += 1
    
    coverage = (states_learned / states_checked * 100) if states_checked > 0 else 0
    print(f"  États libres dans le maze: {states_checked}")
    print(f"  États avec Q-values > 0: {states_learned}")
    print(f"  Couverture: {coverage:.1f}%")
    
    if coverage < 50:
        print(f"❌ PROBLÈME: Couverture trop faible ({coverage:.1f}%)")
        print(f"   → L'agent n'a pas exploré assez d'états")
        print(f"   → Solution: Réentraîner avec plus d'épisodes")
    
    # Recommandations finales
    print("\n" + "="*60)
    print("RECOMMANDATIONS")
    print("="*60)
    
    if coverage < 50:
        print("🔴 PRIORITÉ 1: Réentraîner l'agent")
        print("   python train.py")
        print("   Avec N_EPISODES = 10000")
    elif collision_count > steps * 0.5:
        print("🟠 PRIORITÉ 2: Améliorer l'apprentissage des obstacles")
        print("   1. Augmenter epsilon_decay à 0.998 dans train.py")
        print("   2. Réentraîner avec 5000 épisodes")
    else:
        print("🟡 Ajouter de l'exploration pendant le test")
        print("   Dans test.py, utiliser test_epsilon=0.2")
        print("   results = test_agent(env, agent, test_epsilon=0.2)")


if __name__ == "__main__":
    test_with_detailed_debug()