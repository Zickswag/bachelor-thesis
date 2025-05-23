import argparse
import os

seed = int(os.environ.get('PYTHONHASHSEED', '0'))

# 1) Parse alle Argumente (inkl. seed) direkt am Anfang
parser = argparse.ArgumentParser(description="Train DQN/DDQN on top-down race")
parser.add_argument("--agent",          choices=["dqn","ddqn"], default="dqn")
parser.add_argument("--layers",         nargs="+",              type=int, default=[17,17,17])
parser.add_argument("--lr",             type=float,             default=0.0005)
parser.add_argument("--gamma",          type=float,             default=0.99)
parser.add_argument("--batch_size",     type=int,               default=512)
parser.add_argument("--mem_size",       type=int,               default=25000)
parser.add_argument("--replace",        type=int,               default=50)
parser.add_argument("--save_interval",  type=int,               default=10)
parser.add_argument("--epsilon-start",  type=float,             default=1.0)
parser.add_argument("--epsilon-end",    type=float,             default=0.1)
parser.add_argument("--epsilon-decay",  type=float,             default=0.9995)
parser.add_argument("--render_freq",    type=int,               default=0)
parser.add_argument("--max_steps",      type=int,               default=3000000)
parser.add_argument("--run_name",       type=str,               default=None)
parser.add_argument("--is_pipeline",    action="store_true")
args = parser.parse_args()

import datetime, shutil, json, csv
import pygame
import random
import numpy as np
import tensorflow as tf   # TensorFlow

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from game import game_environment

# Seed setzen
random.seed(seed)           
np.random.seed(seed)        
tf.random.set_seed(seed) 

# Strikten TF-Determinismus aktivieren (TF ≥2.8)
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Konstanten
ACTIONS                 = 5
INPUT_DIMS              = 17
MAX_CHECKPOINT_STEPS    = 100
MAX_EPISODE_STEPS       = 1000
EXPERIMENTS_DIR         = "experiments"
CODE_SNAPSHOT_IGNORE    = ["experiments", "*.pyc", "__pycache__", ".git"]


def setup_run_dir(args):
     # Basisverzeichnis für alle Runs
    base = os.path.join(EXPERIMENTS_DIR, args.agent)

    # Falls Pipeline-Run: timestamp + run_name als einziger Ordner
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S_%f")
    if args.is_pipeline:
        if not args.run_name:
            raise ValueError("Im Pipeline-Modus muss --run_name gesetzt sein.")
        folder = f"{ts}_{args.run_name}"
    else:
        # Normaler Run: nur timestamp
        folder = ts

    run_dir = os.path.join(base, folder)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def snapshot_code(dst):
    shutil.copytree(".", dst, ignore=shutil.ignore_patterns(*CODE_SNAPSHOT_IGNORE))


def main():
    # Run-Ordner anlegen
    run_dir = setup_run_dir(args)

    # Config speichern als JSON
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        config = vars(args).copy()
        config['seed'] = seed
        json.dump(config, f, indent=2)

    # Code-Snapshot
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
    metrics_csv.writerow(["episode","global_steps","score","avg_score","epsilon"])

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
        epsilon_dec=args.epsilon_decay,
        alpha=args.lr,
        gamma=args.gamma,
        replace_target=args.replace,
        mem_size=args.mem_size,
        batch_size=args.batch_size,
        layers=args.layers,
        fname=model_template
)

    scores = []
    eps_history = []
    global_steps = 0
    episode = 0

    # Training
    while global_steps < args.max_steps:
        game.reset()
        episode += 1
        done = False
        score = 0
        checkpoint_steps = 0
        episode_steps = 0
        obs = np.array(game.step(0)[0])
        render_game = (args.render_freq > 0) and (episode % args.render_freq == 0)

        while not done and global_steps < args.max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.choose_action(obs)
            obs_, reward, done = game.step(action)
            obs_ = np.array(obs_)

            # Timeout, wenn kein Reward
            if reward == 0:
                checkpoint_steps += 1
                if checkpoint_steps > MAX_CHECKPOINT_STEPS:
                    done = True
            else:
                checkpoint_steps = 0

            score += reward
            agent.remember(obs, action, reward, obs_, int(done))
            agent.learn()
            obs = obs_

            episode_steps += 1
            global_steps += 1

            if episode_steps > MAX_EPISODE_STEPS:
                done = True

            if render_game:
                game.render(action)

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        # Target-Netzwerk
        if episode % args.replace == 0 and episode > 0:
            agent.update_network_parameters()

        # Modell speichern
        if episode % args.save_interval == 0 and episode > 0:
            agent.save_model(episode)
            print(f"Saved model at episode {episode}")

        # Ausgabe
        print(f"Episode: {episode}, Steps: {global_steps}, Score: {score},"
              f" Average: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

        # TensorBoard-Logging
        with writer.as_default():
            tf.summary.scalar("global_steps", global_steps, step=global_steps)
            tf.summary.scalar("score", score, step=global_steps)
            tf.summary.scalar("avg_score", avg_score, step=global_steps)
            tf.summary.scalar("epsilon", agent.epsilon, step=global_steps)

        # CSV-Log
        metrics_csv.writerow([episode, global_steps, score, avg_score, agent.epsilon])
        metrics_fp.flush()

    metrics_fp.close()


if __name__ == "__main__":
    main()
