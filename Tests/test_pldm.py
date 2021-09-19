import unittest
import sys, os
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Method")
import random
import pldm

class PLDM(unittest.TestCase):

    def test_initMapping(self):
        N = random.randint(2,10)
        qF, qB, pF, pB  = pldm.initMapping(N, 0, "focused")
        self.assertEqual( len(qF) , N)
        self.assertEqual(pB[0] , -1)
        self.assertEqual(pF[1] , 0)
        self.assertEqual(qB[1] , 0)
        qF, qB, pF, pB  = pldm.initMapping(N, 0, "sampled")
        self.assertNotEqual(qB[1] , 0)



if __name__ == '__main__':
    unittest.main()