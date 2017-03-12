import unittest
import numpy as np

from hmm import *

class TestHMM(unittest.TestCase):

  def setUp(self):
    self.model = HMM(['Rain', 'Sun'])
    self.model.update_initial([0.5, 0.5])
    self.model.update_transitions([[0.3, 0.7], [0.1, 0.9]])
    self.model.update_emission('Rain', 'Umbrella', 0.4)
    self.model.update_emission('Rain', 'Coat', 0.6)
    self.model.update_emission('Sun', 'Umbrella', 0.3)
    self.model.update_emission('Sun', 'Sunglasses', 0.7)

  def test_update_transition(self):
    self.model.update_transition('Rain', 'Sun', 0.5)
    iA = self.model._get_state_index('Rain')
    iB = self.model._get_state_index('Sun')
    self.assertEqual(self.model._transition[iA][iB], 0.5)

  def test_update_transition_invalid_prob(self):
    self.assertRaises(AssertionError, self.model.update_transition, 'Rain', 'Sun', 2.0)

  def test_update_transition_invalid_state(self):
    self.assertRaises(AssertionError, self.model.update_transition, 'Snow', 'Sun', 0.1)

  def test_update_transitions(self):
    probs = [[0.2, 0.8], [0.3, 0.7]]
    self.model.update_transitions(probs)
    self.assertTrue(np.all(self.model._transition == probs))

  def test_update_transitions_invalid_shape(self):
    probs = np.array([[0.2, 0.8], [0.3, 0.7], [1.0, 1.0]])
    self.assertRaises(AssertionError, self.model.update_transitions, probs)

  def test_update_emission(self):
    self.model.update_emission('Rain', 'Umbrella', 0.9)
    self.assertEqual(self.model._emission['Rain']['Umbrella'], 0.9)

  def test_update_initial(self):
    self.model.update_initial([0.1, 0.9])
    self.assertTrue(np.all(self.model._initial == [0.1, 0.9]))

  def test_update_initial_invalid_shape(self):
    self.assertRaises(AssertionError, self.model.update_initial, [0.1, 0.2, 0.7])

  def test_forward(self):
    obs = ['Umbrella', 'Coat', 'Sunglasses']
    prob = self.model.forward(obs)
    self.assertAlmostEqual(prob, 0.02205)


if __name__=='__main__':
  unittest.main()