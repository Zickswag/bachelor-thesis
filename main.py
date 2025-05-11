import pygame
import numpy as np
import tensorflow as tf
from agents.dqn_agent import DQNAgent
from game import game_environment

# Trainings-Settings 
MAX_EPISODE_STEPS           = 1000   # max Schritte pro Episode
NUM_EPISODES                = 10000  # Anzahl Episoden
REPLACE_TARGET              = 50     # Episoden bis Target-Netz Update
SAVE_INTERVAL               = 10     # Episoden-Intervall zum Speichern
MAX_CHECKPOINT_STEPS        = 100    # Schritte ohne Reward → Abbruch
RENDER_INTERVAL             = 10     # jede n-te Episode rendern
LOG_DIR                     = "logs/dqn/run"

# TensorBoard Summary Writer 
writer = tf.summary.create_file_writer(LOG_DIR)

game = game_environment.GameEnvironment()
game.fps = 60

# alpha = Lernrate (wie stark Netzwerkgewichte bei jedem Update angepasst werden)
# gamma = Discountfaktor (wie stark wird zukünftige Belohnung gewichtet)
# n_actions = Anzahl der möglichen Aktionen
# epsilon = epsilon greedy (wie oft wird eine zufällige Aktion gewählt)
# epsilon_end = epsilon greedy am Ende (Minimalwert für zufällige Aktionen)
# epsilon_dec = epsilon decay (wie schnell wird epsilon reduziert)
# replace_target = wie oft wird das Target Netzwerk aktualisiert
agent = DQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target=REPLACE_TARGET, mem_size=25000, batch_size=512, input_dims=17, fname='dqn_model.keras')

scores = []
eps_history = []
global_steps = 0

def run():
    global global_steps
    for episode in range(NUM_EPISODES):
        game.reset() 
        done = False
        score = 0
        checkpoint_steps = 0
        episode_steps = 0 
        
        observation = np.array(game.step(0)[0])
        render_game = (episode % RENDER_INTERVAL == 0 and episode > 0)

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    pygame.quit()
                    return
                
            action = agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # Timeout falls zu lange kein Reward            
            if reward == 0:
                checkpoint_steps += 1
                if checkpoint_steps > MAX_CHECKPOINT_STEPS:
                    done = True
            else:
                checkpoint_steps = 0

            score += reward
            agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()
            episode_steps += 1
            global_steps += 1  

            if episode_steps > MAX_EPISODE_STEPS:
                done = True

            if render_game:
                game.render(action)

        # Metriken speichern
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        # Target-Netzwerk periodisch aktualisieren
        if episode % REPLACE_TARGET == 0 and episode > REPLACE_TARGET:
            agent.update_network_parameters()
 
        # Modell speichern
        if episode % SAVE_INTERVAL == 0 and episode > 10:
            agent.save_model()
            print("save model")

        # Konsolenausgabe    
        print(f"Episode: {episode}, Steps: {global_steps}, Score: {score:.2f}, average score: {avg_score:.2f}, epsilon: {agent.epsilon:.2f}, memory size: {agent.memory.mem_cntr % agent.memory.mem_size}")

        # TensorBoard
        with writer.as_default():
            tf.summary.scalar("global_steps", global_steps, step=global_steps)
            tf.summary.scalar("episode_score", score, step=global_steps)
            tf.summary.scalar("average_score", avg_score, step=global_steps)
            tf.summary.scalar("epsilon", agent.epsilon, step=global_steps)

run()        
