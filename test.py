import unittest
from main import *

class Test(unittest.TestCase):
    def __init__(self, *a, **kw):
        self.data_kube = np.array(load_images("data/kube"))
        self.data_ball = np.array(load_images("data/ball"))
        super().__init__(*a, **kw)

    def test_generalCase(self):
        """based on PCA algorithm"""
        ball_vector = generalCase(self.data_ball[:4])
        kube_vector = generalCase(self.data_kube[4:])
        self.assertEqual(comparison(kube_vector.reshape((1, -1)), ball_vector.reshape((1, -1)),
                    self.data_kube[0]), "квадрат")
        self.assertEqual(comparison(kube_vector.reshape((1, -1)), ball_vector.reshape((1, -1)),
                    self.data_ball[0]), "круг")

    def test_Morphological(self):
        """based on morphological method"""
        self.assertEqual(plausibility(self.data_kube[0], self.data_ball[0]), False)
        self.assertEqual(plausibility(self.data_kube[0], self.data_kube[1]), True)
        self.assertEqual(plausibility(self.data_kube[2], self.data_kube[3]), True)

unittest.main(argv=['first-arg-is-ignored'], exit=False)