import pandas as pd

class Text:

    def __init__(self, path):
        self._data = pd.read_csv(path, delim_whitespace=True,
                usecols=[0, 1],
                header=None)
        self._data.columns = ['word', 'tag']

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
        pairs = pd.Series(zip(tags[:tags.size-1], tags[1:]))
        freq = pairs.value_counts()
        return freq[(from_tag, to_tag)]


class TagSet():

    def __init__(self, path):
        self._tags = pd.read_csv(path, sep=';', header=None)
        self._tags.columns = ['tag', 'description']
        self._tags['description'] = self._tags['description'].map(str.strip)

    def describe(self, tag):
        filtered = self._tags.loc[self._tags['tag'] == tag]
        return filtered['description'].values
