import numpy as np
import hmm
import data
import pickle

class POSTagger():

    def __init__(self):
        self._model = None

    def fit(self, text):
        tagset, tag_index = text.get_tagset()
        self._model = hmm.HMM(tagset, tag_index)

        transition_probs = text.calculate_transition_matrix()
        self._model.set_all_transitions(transition_probs)
        for tag in tagset:
            emission = text.count_emission(tag)
            self._model.set_emission(tag, emission)

        initials = text.calculate_initial_probability()
        self._model.set_initial(initials)

    def save_model(self, file_name):
        if self._model is not None:
            with open(file_name, 'wb') as f:
                pickle.dump(self._model, f)

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            self._model = pickle.load(f)

    def predict(self, text):
        return self._model.viterbi(text) if self._model is not None else None

