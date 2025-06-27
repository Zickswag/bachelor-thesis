import numpy as np
import tensorflow 
from game.game_environment import GameEnvironment

# Modellpfad
MODEL_PATH = 'model_ep02200.keras'

# Laden des trainierten Modells
model = tensorflow.keras.models.load_model(MODEL_PATH)

# Umgebung initialisieren
env = GameEnvironment()
state = env.car.cast(env.walls)

done = False

# Hauptspiel-Schleife
while not done:
    env.render(0)

    # Vorbereitung des Zustands für das Modell
    input_state = np.array(state).reshape(1, -1)

    # Modellentscheidung (beste Aktion basierend auf Q-Werten)
    q_values = model.predict(input_state, verbose=0)
    action = np.argmax(q_values[0])  # Wähle Aktion mit höchstem Q-Wert

    # Schritt ausführen
    state, reward, done = env.step(action)

if done:
    env.reset()
    state = env.car.cast(env.walls)
    done = False

# Spiel beenden
env.close()
