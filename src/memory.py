class Memory():
    def __init__(self, num_envs):
        self.num_envs = num_envs

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.rnd_current_states = []
        self.rnd_next_states = []
        self.rnd_current_states_norm = []
        self.rnd_next_states_norm = []
        self.rewards_int = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def save(self, state, action, reward, rnd_current_state, rnd_next_state, rnd_current_state_norm, rnd_next_state_norm,
             reward_int, next_state, done, logit, value, value_int):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.rnd_current_states.append(rnd_current_state)
        self.rnd_next_states.append(rnd_next_state)
        self.rnd_current_states_norm.append(rnd_current_state_norm)
        self.rnd_next_states_norm.append(rnd_next_state_norm)
        self.rewards_int.append(reward_int)
        self.dones.append(done)
        self.logits.append(logit)
        self.values.append(value)
        self.values_int.append(value_int)

    def reset(self):
        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.rnd_current_states = []
        self.rnd_next_states = []
        self.rnd_current_states_norm = []
        self.rnd_next_states_norm = []
        self.rewards_int = []
        self.dones = []
        self.logits = []
        self.values = []
        self.values_int = []

    def get_data(self):
        return self.states, self.actions, self.next_states, self.rewards, self.rnd_current_states, self.rnd_next_states, \
                self.rnd_current_states_norm, self.rnd_next_states_norm, self.rewards_int, self.dones, self.logits, self.values, self.values_int