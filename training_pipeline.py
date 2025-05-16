"""
Führt nacheinander mehrere Trainingsläufe mit verschiedenen
Kombinationen von Parametern für main.py aus,
inkl. Seed- und Hash-Seed-Handling.
"""
import os
import subprocess
import itertools

# Parameterbereiche definieren
learning_rate   = [0.0001, 0.0005, 0.001]
gamma           = [0.99]
layer_set       = [[17], [17, 17], [17, 17, 17], [17, 17, 17, 17], [17, 17, 17, 17, 17]]
agent           = ["dqn"]
batch_size      = [512]
mem_size        = [25000]
replace_target  = [50]
save_interval   = [10]
epsilon_start   = [1.0]
epsilon_end     = [0.1]
epsilon_decay   = [0.9995]
render_freq     = [0]
max_steps       = [5000]
seed            = [0]

# Erzeuge das kartesische Produkt aller Kombinationen
experiments = list(itertools.product(
    agent,
    layer_set,
    learning_rate,
    gamma,
    batch_size,
    mem_size,
    replace_target,
    save_interval,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    render_freq,
    max_steps,      
    seed
))


for idx, (agent, layer_set, learning_rate, gamma, batch_size, mem_size, replace_target, save_interval, epsilon_start, epsilon_end, epsilon_decay, render_freq, max_steps, seed) in enumerate(experiments, start=1):
    # Parameter-String für Verzeichnisname (vollständige Parameter)
    layers_str = "x".join(str(l) for l in layer_set)
    param_str = (
        f"run{idx:03d}_{agent}_lay{layers_str}_lr{learning_rate}_g{gamma}_"
        f"bs{batch_size}_mem{mem_size}_rep{replace_target}_save{save_interval}_"
        f"eps{epsilon_start}-{epsilon_end}-d{epsilon_decay}_"
        f"rf{render_freq}_ms{max_steps}_s{seed}"
    )

    # 1) Setze Environment-Variable für Python-Hash-Seed und Output-Dir
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)

    # 2) Baue die Kommandozeile für main.py
    cmd = [
        "python", "main.py",
        "--agent", agent,
        "--lr", str(learning_rate),
        "--gamma", str(gamma),
        "--layers", *map(str, layer_set),
        "--epsilon-start", str(epsilon_start),
        "--epsilon-end", str(epsilon_end),
        "--epsilon-decay", str(epsilon_decay),
        "--max_steps", str(max_steps),
        "--batch_size", str(batch_size),
        "--mem_size", str(mem_size),
        "--replace", str(replace_target),
        "--save_interval", str(save_interval),
        "--render_freq", str(render_freq),
        "--run_name", str(param_str),
        "--is_pipeline"
    ]

    print(f"==> Experiment {idx}/{len(experiments)}: {param_str} ")

    # 3) Starte das Training
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    # 4) Ausgabe der Logs
    print(result.stdout)
    if result.returncode != 0:
        print("FEHLER:", result.stderr)
        # Optional: Abbrechen oder fortsetzen
        # break

print("Alle Experimente abgeschlossen.")
