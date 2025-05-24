import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
    """
    ReplayBuffer speichert Erfahrungen (state, action, reward, next_state, done)
    und liefert zufällige Samples für das Training (Experience Replay).
    """
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        """
        Speichert einen Übergang in den Puffer.
        Ältere Übergänge werden im ring buffer Modus überschrieben.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        #self.terminal_memory[index] = 1 - done
        self.terminal_memory[index] = 0.0 if done else 1.0
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Zieht eine Zufallsstichprobe aus dem Replay Buffer.
        Liefert Tupel (states, actions, rewards, next_states, dones).
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal


class DQNAgent(object):
    """
    Deep Q-Network Agent mit Target-Network, Epsilon-Greedy Exploration
    und Experience Replay.
    """
    def __init__(self, alpha, gamma, actions, epsilon, batch_size, input_dims, epsilon_dec, epsilon_end, mem_size, replace_target, layers, fname):
        self.action_space = [i for i in range(actions)]
        self.n_actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, actions, discrete=True)
        self.layers = layers

        # Eval- und Zielnetzwerk (Target Network)        
        self.eval_network = QNetwork(input_dims, actions, batch_size, alpha, layers)
        self.target_network = QNetwork(input_dims, actions, batch_size, alpha, layers)
        self.target_network.copy_weights(self.eval_network)

        
    def remember(self, state, action, reward, new_state, done):
        """Speichere die Erfahrung im Replay Buffer"""
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        """
        Wähle Aktion per Epsilon-Greedy:
         - Zufall mit Wahrscheinlichkeit ε
         - sonst: argmax_a Q_eval(state, a)
        """
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.eval_network.q_eval(state_tensor)
        action = int(tf.argmax(q_values[0]))
        return action
    
    def learn(self):
        """
        Ziehe Batch aus Replay Buffer, berechne Ziel-Q-Werte,
        und führe einen Trainingsschritt auf dem Eval-Netzwerk aus.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample aus Replay Buffer
        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)

        # Tensor-Conversion
        states_tensor  = tf.convert_to_tensor(states, dtype=tf.float32)
        next_tensor    = tf.convert_to_tensor(states_, dtype=tf.float32)

        # Graph-Eval beider Netze
        q_next = self.target_network.q_eval(next_tensor)
        q_pred = self.eval_network.q_eval(states_tensor)

        # Ziel-Q-Werte in NumPy berechnen
        q_target = q_pred.numpy()
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        action_values = np.array(self.action_space, dtype=np.int32)
        action_idx = np.dot(actions, action_values)

        q_target[batch_idx, action_idx] = (rewards + self.gamma * np.max(q_next.numpy(), axis=1) * dones)

        # Clipping zur Sicherheit
        q_target = np.clip(q_target, -1e3, 1e3)

        # Training + NaN-Prüfung
        self.eval_network.train_step(states_tensor, tf.convert_to_tensor(q_target, dtype=tf.float32))

        # Epsilon-Decay
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min


    def update_network_parameters(self):
        """Kopiere Gewichte vom Eval- ins Target-Netzwerk"""
        self.target_network.copy_weights(self.eval_network)

    def save_model(self, episode: int):
        """Speichere das Eval-Modell auf Festplatte"""
        path = self.model_file.format(episode=episode)
        self.eval_network.model.save(path)
        
    def load_model(self):
        """Lade Modell und gleiche Target-Netzwerk an"""
        self.eval_network.model = tf.keras.models.load_model(self.model_file)
        self.target_network.model = tf.keras.models.load_model(self.model_file)
        if self.epsilon == 0.0:
            self.update_network_parameters()


class QNetwork:
    """
    Keras-basiertes Feedforward-Netzwerk für Q-Funktionsapproximation.
    """
    def __init__(self, NbrStates, NbrActions, batch_size, lr, layers):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.layers = layers
        self.model = self.create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    
    def create_model(self):
        """Erzeuge Sequential-Modell mit gegebener Layer-Architektur"""
        model_layers = []
        for units in self.layers:
            model_layers.append(tf.keras.layers.Dense(units, activation=tf.nn.relu))
        model_layers.append(tf.keras.layers.Dense(self.NbrActions))
        return tf.keras.Sequential(model_layers)
    
    def copy_weights(self, source_net):
        """Kopiere trainierbare Variablen von einer Quelle"""
        variables1 = self.model.trainable_variables
        variables2 = source_net.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
            
    @tf.function
    def train_step(self, states, q_targets):
        """Einzelner Gradientenschritt auf Basis MeanSquaredError"""
        with tf.GradientTape() as tape:
            q_pred = self.model(states, training=True)
            loss   = self.loss_fn(q_targets, q_pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return loss

    # Graph-Eval-Funktion
    @tf.function
    def q_eval(self, states):
        """Q-Werte-Prediction (Inference-Modus)"""
        return self.model(states, training=False)
    
        