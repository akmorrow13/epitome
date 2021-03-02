from epitome.test import EpitomeTestCase
from epitome.test import *
from epitome.functions import *
from epitome.dataset import *
import pytest
import warnings


class FunctionsTest(EpitomeTestCase):

    def __init__(self, *args, **kwargs):
        super(FunctionsTest, self).__init__(*args, **kwargs)

    def test_user_data_path(self):
        # user data path should be able to be explicitly set
        datapath = GET_DATA_PATH()
        assert(datapath == os.environ["EPITOME_DATA_PATH"])


    def test_compute_casv(self):

        m1 = np.zeros((3, 1))
        m2 = np.zeros((3, 1))

        casv = compute_casv(m1, m2, radii=[1,3])
        assert(casv.shape == (3, 4))
