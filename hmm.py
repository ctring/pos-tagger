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

    def update_initial(self, init):
        init = np.array(init)
        assert init.shape == (self._N,), 'Shape must be {}'.format((self._N,))
        assert np.isclose(np.sum(init), 1.0), 'Sum of all probabilities must be 1.0'
        self._initial = np.copy(init)

    def update_transition(self, from_state, to_state, prob):
        i_from = self._get_state_index(from_state)
        i_to = self._get_state_index(to_state)
        assert 0.0 <= prob <= 1.0, 'Probability value must lie between 0.0 and 1.0'
        self._transition[i_from][i_to] = prob

    def update_transitions(self, transition_probs):
        transition_probs = np.array(transition_probs)
        assert transition_probs.shape == (self._N, self._N), 'Shape must be {}'.format((self._N, self._N))
        row_sum = np.sum(transition_probs, axis=1)
        assert np.all(np.isclose(row_sum, 1.0)), 'Sum of each row must be 1.0'
        assert np.all(0 <= transition_probs) and np.all(transition_probs <= 1.0), 'All values must lie between 0.0 and 1.0'
        self._transition = np.copy(transition_probs)

    def update_emission(self, state, obs, prob):
        assert 0.0 <= prob <= 1.0, 'Probability value must lie between 0.0 and 1.0'
        assert state in self._emission, 'State "{}" is not defined'.format(state)
        self._emission[state][obs] = prob

    def forward(self, obs):
        acc = np.empty((2, self._N))
        for s in self._states:
            i = self._state_index[s]
            acc[1][i] = self._initial[i] * self._emission[s].get(obs[0], 0)

        for i, o in enumerate(obs[1:]):
            acc[i % 2] = np.dot(acc[1 - i % 2], self._transition)
            for j, s in enumerate(self._states):
                acc[i % 2][j] *= self._emission[s].get(o, 0)

        prob = np.sum(acc[len(obs) % 2])
        return prob

#     def viterbi(self, obs):
#         v = np.zeros((2, self.N))
#         bt = np.ones((2, self.N), dtype=np.int)
#         v[0] = self.pS * self.B[:, obs[0]]
#         for i, o in enumerate(obs[1:]):
#             cur = 1 - i % 2
#             prev = i % 2
#             for s, vp in enumerate(v[prev]):
#                 v[cur, s] = np.max(v[prev] * self.A[:, s]) * self.B[s, o]
#                 bt[cur, s] = np.argmax(v[prev] * self.A[:, s])
#         prob = np.max(v[1 - len(obs) % 2])
#         states = np.zeros(self.N)
#         s = np.argmax(v[1 - len(obs) % 2])
#         for i in range(len(obs) - 1, -1, -1):
#             states[i] = s
#             s = bt[i, s]
#         return prob, states


