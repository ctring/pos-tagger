import pandas as pd
import numpy as np

SPACE_TAG = 'SPACE'

class Text:

    def __init__(self, path):
        self._data = pd.read_csv(path, delim_whitespace=True,
                usecols=[0, 1], header=None, skip_blank_lines=False)
        self._data.columns = ['word', 'tag']
        self._data['tag'] = self._data['tag'].fillna(SPACE_TAG)

        self._cached_tagset = None
        self._cached_tag_index = None
        self._cached_transition_matrix = None

    def calculate_transition_matrix(self):
        if self._cached_transition_matrix is None:
            tagset, tag_index = self.get_tagset()

            N = len(tagset)
            probs = np.zeros((N, N), dtype=np.float)

            tag_data = self._data['tag']
            for i in range(tag_data.size - 1):
                current_tag_ix = tag_index[tag_data.iloc[i]]
                next_tag_ix = tag_index[tag_data.iloc[i + 1]]
                probs[current_tag_ix, next_tag_ix] += 1

            probs = probs / np.sum(probs, axis=1, keepdims=True)
            self._cached_transition_matrix = probs
        return self._cached_transition_matrix

    def calculate_initial_probability(self):
        tagset, tag_index = self.get_tagset()
        probs = self.calculate_transition_matrix()
        return probs[tag_index[SPACE_TAG]]

    def count_transition(self, from_tag=None, to_tag=None):
        if from_tag is None and to_tag is None:
            return 0
        tags = self._data['tag']
        if to_tag is None:
            freq = tags[:tags.size-1].value_counts()
            return freq[from_tag] if from_tag in freq else 0
        if from_tag is None:
            freq = tags[1:].value_counts()
            return freq[to_tag] if to_tag in freq else 0
        pairs = pd.Series(list(zip(tags[:tags.size-1], tags[1:])))
        freq = pairs.value_counts()
        return freq.get((from_tag, to_tag), 0)

    def count_emission(self, from_tag, to_word=None):
        filtered = self._data[self._data['tag'] == from_tag]
        if to_word:
            return filtered[filtered['word'] == to_word].shape[0]
        word_prob = filtered['word'].value_counts(normalize=True)
        return word_prob.to_dict()

    def get_words(self):
        return self._data['word'].tolist()

    def get_tagset(self):
        if not self._cached_tagset:
            self._cached_tagset = self._data['tag'].unique().tolist()
            self._cached_tag_index = {tag:i for i, tag in enumerate(self._cached_tagset)}
        return self._cached_tagset, self._cached_tag_index
