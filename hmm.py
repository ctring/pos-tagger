'''Hidden Markov Model'''
import numpy as np

class HMM():

    def __init__(self, states):
        self._states = list(states)
        self._N = len(states)
        self._state_index = {s:i for i, s in enumerate(self._states)}
        self._transition = np.zeros((self._N, self._N))
        self._emission = {s:{} for s in self._states}
        self._initial = np.zeros(self._N)

    def _get_state_index(self, state):
        i = self._state_index.get(state, -1)
        assert i >= 0, 'State "{}" is not defined'.format(state)
        return i

    def set_initial(self, init):
        init = np.array(init)
        assert init.shape == (self._N,), 'Shape must be {}'.format((self._N,))
        assert np.isclose(np.sum(init), 1.0), 'Sum of all probabilities must be 1.0'
        self._initial = np.copy(init)

    def set_transition(self, from_state, to_state, prob):
        i_from = self._get_state_index(from_state)
        i_to = self._get_state_index(to_state)
        assert 0.0 <= prob <= 1.0, 'Probability value must lie between 0.0 and 1.0'
        self._transition[i_from][i_to] = prob

    def set_all_transitions(self, transition_probs):
        transition_probs = np.array(transition_probs)
        assert transition_probs.shape == (self._N, self._N), 'Shape must be {}'.format((self._N, self._N))
        row_sum = np.sum(transition_probs, axis=1)
        assert np.all(row_sum <= 1.0), 'Sum of each row must be smaller than 1.0'
        assert np.all(0 <= transition_probs) and np.all(transition_probs <= 1.0), 'All values must lie between 0.0 and 1.0'
        self._transition = np.copy(transition_probs)

    def set_emission(self, state, obs):
        assert isinstance(obs, dict), 'Emission probabilities must be a dict'
        self._emission[state] = obs.copy()

    def forward(self, obs):
        N_obs = len(obs)
        acc = np.zeros((2, self._N))
        for j, s in enumerate(self._states):
            acc[0][j] = self._initial[j] * self._emission[s].get(obs[0], 0)

        for i in range(1, N_obs):
            acc[i % 2] = np.dot(acc[1 - i % 2], self._transition)
            for j, s in enumerate(self._states):
                acc[i % 2][j] *= self._emission[s].get(obs[i], 0)

        prob = np.sum(acc[(N_obs - 1) % 2])
        return prob

    def viterbi(self, obs):
        N_obs = len(obs)
        acc = np.zeros((2, self._N))
        trace = np.zeros((N_obs, self._N), dtype=np.int)
        for j, s in enumerate(self._states):
            acc[0][j] = self._initial[j] * self._emission[s].get(obs[0], 0)

        for i in range(1, N_obs):
            for j, s in enumerate(self._states):
                pred = acc[1 - i % 2] * self._transition[:, j]
                acc[i % 2, j] = np.max(pred) * self._emission[s].get(obs[i], 0)
                trace[i, j] = np.argmax(pred)

        prob = np.max(acc[(N_obs - 1) % 2])
        s = np.argmax(acc[(N_obs - 1) % 2])
        states = []
        for i in range(N_obs - 1, -1, -1):
            states.append(self._states[s])
            s = trace[i, s]
        return prob, states[::-1]

