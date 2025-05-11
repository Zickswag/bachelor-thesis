import numpy as np
import tensorflow as tf

class ReplayBuffer(object):
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
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal


class DQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec, epsilon_end,
                 mem_size, replace_target, fname):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
       
        self.eval_network = QNetwork(input_dims, n_actions, batch_size, lr=alpha)
        self.target_network = QNetwork(input_dims, n_actions, batch_size, lr=alpha)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.eval_network.q_eval(state_tensor)
        action = int(tf.argmax(q_values[0]))
        return action

    def learn(self):
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
        # Aktion-Indices berechnen
        action_values = np.array(self.action_space, dtype=np.int32)
        action_idx = np.dot(actions, action_values)
        # Q-Ziel setzen
        q_target[batch_idx, action_idx] = (
            rewards + self.gamma * np.max(q_next.numpy(), axis=1) * dones
        )

        # Graph-Trainings-Schritt via tf.function
        self.eval_network.train_step(states_tensor, tf.convert_to_tensor(q_target, dtype=tf.float32))

        # Epsilon-Decay
        # self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        

    def update_network_parameters(self):
        self.target_network.copy_weights(self.eval_network)

    def save_model(self):
        self.eval_network.model.save(self.model_file)
        
    def load_model(self):
        self.eval_network.model = tf.keras.models.load_model(self.model_file)
        self.target_network.model = tf.keras.models.load_model(self.model_file)
        if self.epsilon == 0.0:
            self.update_network_parameters()


class QNetwork:
    def __init__(self, NbrStates, NbrActions, batch_size, lr):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.model     = self.createModel()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn   = tf.keras.losses.MeanSquaredError()
    
    def createModel(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.NbrActions)
        ])
        return model
    
    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
            
    @tf.function
    def train_step(self, states, q_targets):
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
        return self.model(states, training=False)
    
        