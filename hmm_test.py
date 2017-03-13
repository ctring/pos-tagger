import unittest
import numpy as np

from hmm import *

class TestHMM(unittest.TestCase):

  def setUp(self):
    self.model = HMM(['Rain', 'Sun'])
    self.model.set_initial([0.5, 0.5])
    self.model.set_all_transitions([[0.3, 0.7], [0.1, 0.9]])
    self.model.set_emission('Rain', {'Umbrella': 0.4, 'Coat': 0.6})
    self.model.set_emission('Sun', {'Umbrella': 0.3, 'Sunglasses': 0.7})

  def test_set_transition(self):
    self.model.set_transition('Rain', 'Sun', 0.5)
    iA = self.model._get_state_index('Rain')
    iB = self.model._get_state_index('Sun')
    self.assertEqual(self.model._transition[iA][iB], 0.5)

  def test_set_transition_invalid_prob(self):
    self.assertRaises(AssertionError, self.model.set_transition, 'Rain', 'Sun', 2.0)

  def test_set_transition_invalid_state(self):
    self.assertRaises(AssertionError, self.model.set_transition, 'Snow', 'Sun', 0.1)

  def test_set_all_transitions(self):
    probs = [[0.2, 0.8], [0.3, 0.7]]
    self.model.set_all_transitions(probs)
    self.assertTrue(np.all(self.model._transition == probs))

  def test_set_all_transitions_invalid_shape(self):
    probs = np.array([[0.2, 0.8], [0.3, 0.7], [1.0, 1.0]])
    self.assertRaises(AssertionError, self.model.set_all_transitions, probs)

  def test_set_emission(self):
    self.model.set_emission('Rain', {'Umbrella': 0.9})
    self.assertEqual(self.model._emission['Rain']['Umbrella'], 0.9)

  def test_set_emission_invalid(self):
    self.assertRaises(AssertionError, self.model.set_emission, 'Rain', [0.9])

  def test_set_initial(self):
    self.model.set_initial([0.1, 0.9])
    self.assertTrue(np.all(self.model._initial == [0.1, 0.9]))

  def test_set_initial_invalid_shape(self):
    self.assertRaises(AssertionError, self.model.set_initial, [0.1, 0.2, 0.7])

  def test_forward(self):
    obs = ['Umbrella', 'Coat', 'Sunglasses']
    prob = self.model.forward(obs)
    self.assertAlmostEqual(prob, 0.02205)

  def test_viterbi(self):
    obs = ['Umbrella', 'Coat', 'Sunglasses']
    prob, states = self.model.viterbi(obs)
    self.assertAlmostEqual(prob, 0.01764)
    self.assertSequenceEqual(states, ['Rain', 'Rain', 'Sun'])

  def test_viterbi_2(self):
    obs = ['Umbrella', 'Umbrella', 'Umbrella']
    prob, states = self.model.viterbi(obs)
    self.assertAlmostEqual(prob, 0.01134)
    self.assertSequenceEqual(states, ['Rain', 'Sun', 'Sun'])


if __name__=='__main__':
  unittest.main()