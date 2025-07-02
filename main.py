import argparse
import os

# Hole Seed aus Umgebungsvariable oder setze auf 0, falls nicht definiert (für Reproduzierbarkeit)
seed = int(os.environ.get('PYTHONHASHSEED', '0'))

# Argumente definieren (Hyperparameter etc.)
parser = argparse.ArgumentParser(description="DQN vs DDQN Training")
parser.add_argument("--agent",                  choices=["dqn","ddqn"], default="dqn")
parser.add_argument("--layers",                 nargs="+",              type=int, default=[18,18,18])
parser.add_argument("--lr",                     type=float,             default=0.0001)
parser.add_argument("--gamma",                  type=float,             default=0.99)
parser.add_argument("--batch_size",             type=int,               default=512)
parser.add_argument("--mem_size",               type=int,               default=250000)
parser.add_argument("--exploration_steps",      type=int,               default=12500)
parser.add_argument("--replace_target_steps",   type=int,               default=10000)
parser.add_argument("--save_interval",          type=int,               default=100)
parser.add_argument("--epsilon_start",          type=float,             default=1.0)
parser.add_argument("--epsilon_end",            type=float,             default=0.1)
parser.add_argument("--epsilon_decay_steps",    type=int,               default=250000)
parser.add_argument("--render_freq",            type=int,               default=0)
parser.add_argument("--max_steps",              type=int,               default=50000000)
args = parser.parse_args()

# Imports
import datetime, shutil, json, csv
import pygame
import random
import numpy as np
import tensorflow as tf 

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from game import game_environment

# Alle Zufallsquellen deterministisch stellen
random.seed(seed)           
np.random.seed(seed)        
tf.random.set_seed(seed) 

# Konstanten
ACTIONS                 = 5
INPUT_DIMS              = 18
MAX_CHECKPOINT_STEPS    = 100
MAX_EPISODE_STEPS       = 1000
EXPERIMENTS_DIR         = "experiments"
CODE_SNAPSHOT_IGNORE    = ["experiments", "*.pyc", "__pycache__", ".git"]

# Erstelle Ordnerstruktur für Experiment
def setup_run_dir(args):
    base = os.path.join(EXPERIMENTS_DIR, args.agent)
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S_%f")
    run_dir = os.path.join(base, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Speichere aktuellen Code als Snapshot (außer CODE_SNAPSHOT_IGNORE)
def snapshot_code(dst):
    shutil.copytree(".", dst, ignore=shutil.ignore_patterns(*CODE_SNAPSHOT_IGNORE))

# Initialisierung und Training
def main():
    run_dir = setup_run_dir(args)

    # Konfiguration speichern als JSON
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        config = vars(args).copy()
        config['seed'] = seed
        json.dump(config, f, indent=2)

    snapshot_code(os.path.join(run_dir, "code_snapshot"))

    # TensorBoard
    tb_dir = os.path.join(run_dir, "tb_logs")
    writer = tf.summary.create_file_writer(tb_dir)

    # Checkpoints
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_template = os.path.join(ckpt_dir, "model_ep{episode:05d}.keras")

    # CSV-Metriken
    metrics_fp  = open(os.path.join(run_dir, "metrics.csv"), "w", newline="")
    metrics_csv = csv.writer(metrics_fp)
    metrics_csv.writerow(["episode","global_steps","score","avg_score","epsilon","avg_max_q"])

    # Spiel-Umgebung
    game = game_environment.GameEnvironment()

    # Agent instanziieren
    if args.agent == "dqn":
        AgentClass = DQNAgent
    else:
        AgentClass = DDQNAgent

    agent = AgentClass(
        actions=ACTIONS,
        input_dims=INPUT_DIMS,
        epsilon=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        alpha=args.lr,
        gamma=args.gamma,
        replace_target_steps=args.replace_target_steps,
        mem_size=args.mem_size,
        batch_size=args.batch_size,
        layers=args.layers,
        fname=model_template
    )

    scores = []
    eps_history = []
    global_steps = 0
    episode = 0

    # Trainingsloop
    while global_steps < args.max_steps:
        game.reset()
        episode += 1
        done = False
        score = 0
        checkpoint_steps = 0
        episode_steps = 0
        state = np.array(game.step(0)[0])
        render_game = (args.render_freq > 0) and (episode % args.render_freq == 0)

        while not done and global_steps < args.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Exploration oder Policy-Aktion
            action = random.choice(agent.action_space) if global_steps < args.exploration_steps else agent.choose_action(state)

            next_state, reward, done = game.step(action)

            # Fehlerbehandlung bei next_state
            if next_state is None:
                next_state = np.zeros(INPUT_DIMS, dtype=np.float32) if done else None
                valid_transition = done
            else:
                next_state = np.array(next_state, dtype=np.float32)
                # Datentyp überprüpfung und Formvalidierung
                valid_transition = (
                    isinstance(state, np.ndarray) and isinstance(next_state, np.ndarray)
                    and state.dtype == np.float32 and next_state.dtype == np.float32
                    and state.shape == (INPUT_DIMS,) and next_state.shape == (INPUT_DIMS,)
                    and not np.isnan(state).any() and not np.isnan(next_state).any()
                )

            # Beenden bei Inaktivität (kein Checkpoint erreicht)
            checkpoint_steps = 0 if reward != 0 else checkpoint_steps + 1
            if checkpoint_steps > MAX_CHECKPOINT_STEPS:
                done = True

            score += reward

            # Speichern & Lernen nur bei gültiger Beobachtung
            if valid_transition:
                # fülle Replay-Buffer mit gültiger Transition
                agent.remember(state, action, reward, next_state, int(done))
                if global_steps >= args.exploration_steps:
                    current_avg_max_q = agent.learn()
                else: # Während der initialen Exploration nicht lernen
                    current_avg_max_q = None
                state = next_state

            episode_steps += 1
            global_steps += 1

            # Beenden nach maximalen Schritten pro Episode
            if episode_steps > MAX_EPISODE_STEPS:
                done = True

            # Rendern des Spiels
            if render_game:
                game.render(action, episode, global_steps, current_avg_max_q, episode_steps)

        # Nach der Episode
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        # Modell speichern
        if episode % args.save_interval == 0 and episode > 0:
            agent.save_model(episode)
            print(f"Saved model at episode {episode}")

        q_value_str = f"{current_avg_max_q:.2f}" if current_avg_max_q is not None else "N/A"

        # Ausgabe
        print(f"Episode: {episode},"
              f" Steps: {global_steps}, Score: {score},"
              f" Average: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f},"
              f" Length: {episode_steps}, Avg Max Q: {q_value_str}")

        # TensorBoard-Logging
        with writer.as_default():
            tf.summary.scalar("global_steps", global_steps, step=global_steps)
            tf.summary.scalar("episode_length", episode_steps, step=global_steps)
            tf.summary.scalar("score", score, step=global_steps)
            tf.summary.scalar("avg_score", avg_score, step=global_steps)
            tf.summary.scalar("epsilon", agent.epsilon, step=global_steps)
            if current_avg_max_q is not None: # Nur loggen, wenn gelernt wurde
                tf.summary.scalar("avg_max_q_estimate", current_avg_max_q, step=global_steps)

        # CSV-Log
        metrics_csv.writerow([episode, episode_steps, global_steps, score, avg_score, agent.epsilon, current_avg_max_q if current_avg_max_q is not None else np.nan])
        metrics_fp.flush()

    metrics_fp.close()


if __name__ == "__main__":
    main()
