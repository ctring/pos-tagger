import pandas as pd
import unittest

from data import *

_DATA_PATH = 'data/train.txt'
_TAGSET_PATH = 'data/tags.txt'

class TestText(unittest.TestCase):
    def setUpClass(self):
        self.data = Text(_DATA_PATH)

    def test_count_from_only_transition(self):
        count = self.data.count_transition(from_tag='NN')
        self.assertEqual(count, 30147)

    def test_count_to_only_transition(self):
        count = self.data.count_transition(to_tag='NN')
        self.assertEqual(count, 30146)

    def test_count_none_transition(self):
        self.assertEqual(self.data.count_transition(), 0)

    def test_count_transition(self):
        count = self.data.count_transition(from_tag='NN', to_tag='VB')
        self.assertEqual(count, 43)


class TestTagSet(unittest.TestCase):
    def setUpClass(self):
        self.tagset = TagSet(_TAGSET_PATH)

    def test_description(self):
        desc = self.tagset.describe('JJ')
        self.assertEqual(desc, 'Adjective')


if __name__=='__main__':
    unittest.main()
