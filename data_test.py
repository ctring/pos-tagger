import pandas as pd
import unittest

from data import *

_DATA_PATH = 'data/train.txt'
# _TAGSET_PATH = 'data/tags.txt'

class TestText(unittest.TestCase):
    def setUp(self):
        self.data = Text(_DATA_PATH)

    # def test_count_from_only_transition(self):
    #     count = self.data.count_transition(from_tag='NN')
    #     self.assertEqual(count, 30147)

    # def test_count_to_only_transition(self):
    #     count = self.data.count_transition(to_tag='NN')
    #     self.assertEqual(count, 30146)

    # def test_count_none_transition(self):
    #     self.assertEqual(self.data.count_transition(), 0)

    # def test_count_transition(self):
    #     count = self.data.count_transition(from_tag='NN', to_tag='VB')
    #     self.assertEqual(count, 43)

    def test_count_emission(self):
        count = self.data.count_emission(from_tag='VB', to_word='defraud')
        self.assertEqual(count, 4)

    def test_count_all_emission(self):
        count_dict = self.data.count_emission(from_tag='CD')
        self.assertEqual(type(count_dict), dict)


# class TestTagSet(unittest.TestCase):
#     def setUp(self):
#         self.tagset = TagSet(_TAGSET_PATH)

#     def test_description(self):
#         desc = self.tagset.describe('JJ')
#         self.assertEqual(desc, 'Adjective')

#     def test_length(self):
#         self.assertTrue(len(self.tagset) > 0)

#     def test_contain(self):
#         self.assertTrue('JJ' in self.tagset)


if __name__=='__main__':
    unittest.main()
