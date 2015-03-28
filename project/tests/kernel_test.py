import unittest
import numpy as np

from lib import Kernel


class TestKernelOps(unittest.TestCase):
    def setUp(self):
        self.kernel = Kernel()

    def test_add(self):
        a = np.random.rand(50000).astype(np.float32)
        b = np.random.rand(50000).astype(np.float32)

        sum_gpu = self.kernel.sum(a,b)
        sum_cpu = a + b

        print sum_gpu

        np.testing.assert_array_equal(sum_gpu, sum_cpu)
