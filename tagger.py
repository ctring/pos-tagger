import numpy as np
import hmm
import data

class POSTagger():

    def __init__(self, tagset):
        self._tags = tagset.get_tags()
        self._model = hmm.HMM(self._tags)

    def fit(self, text):
        N = len(self._tags)
        transition_probs = np.zeros((N, N))
        print('Counting transitions')
        for i, t_from in enumerate(self._tags):
            count_from = text.count_transition(t_from)
            if count_from == 0:
                continue
            for j, t_to in enumerate(self._tags):
                count_from_to = text.count_transition(t_from, t_to)
                transition_probs[i, j] = float(count_from_to) / count_from
        self._model.set_all_transitions(transition_probs)
        print('Counting emissions')
        for tag in self._tags:
            emission = text.count_emission(tag)
            self._model.set_emission(tag, emission)
        print('Counting initials')
        initials = np.zeros(N)
        for i, tag in enumerate(self._tags):
            initials[i] = text.count_tag(tag)
        self._model.set_initial(initials / np.sum(initials))

    def predict(self, text):
        return self._model.viterbi(text)

