import unittest

import time
import math

from utils import timeSince, report_times

from pdb import set_trace as st

class TestStringMethods(unittest.TestCase):

    def test_report_times(self):
        eps = 0.01
        ##
        start = time.time()
        time.sleep(0.5)
        msg, h = timeSince(start)
        ##
        start = time.time()
        time.sleep(0.5)
        msg, seconds, _, _ = report_times(start)
        ##
        diff = abs(seconds - h*60*60)
        self.assertTrue(diff <= eps)

if __name__ == '__main__':
    unittest.main()
