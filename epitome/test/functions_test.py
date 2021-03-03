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


    def test_get_radius_indices(self):

        indices = get_radius_indices([1,4], 0, 4, 30)
        self.assertEquals(np.all(indices == [4]),True)

        indices = get_radius_indices([1,4], 1, 4, 30)
        self.assertEquals(np.all(indices ==np.array([1, 2, 3, 5, 6, 7])), True)


        indices = get_radius_indices([1,4], 0, 0, 30)
        self.assertEquals(len(indices), 1)

    def test_simple_casv(self):
        nregions = 10
        nassays = 1
        ncells = 1
        nsamples = 1
        radii = [1,3]

        # m1: reference cells
        m1 = np.zeros((nregions, nassays, ncells))

        # m2: samples
        m2 = np.zeros((nregions, nassays, nsamples))

        casv = compute_casv(m1, m2, radii)
        assert(casv.shape == (nregions, nassays*2*len(radii), ncells, nsamples))
        for i in range(casv.shape[0]):
            # first two numbers: pos/pos for r1, r2
            assert casv[i,0,0,0] == 0
            assert casv[i,1,0,0] == 0
            # second two numbers: equal/equal for r1, r2
            assert casv[i,2,0,0] == 1
            assert casv[i,3,0,0] == 1

        casv_noradius =  compute_casv(m1, m2, [])
        assert(casv_noradius.shape ==(10, 0, 1, 1))



    def test_compute_casv(self):
        nregions = 2
        nassays = 1
        ncells = 1
        nsamples = 2
        radii = [1]

        # m1: reference cells
        m1 = np.zeros((nregions, nassays, ncells))
        m1[0]=1 # first region has a signal

        # m2: samples
        m2 = np.zeros((nregions, nassays, nsamples))
        m2[0,:,0] = 1 # first cell has 1 signal in region 1
        m2[:,:,1] = 1 # second cell has 2 signals in both regions
        casv = compute_casv(m1, m2, radii=radii)

        assert(casv.shape == (nregions, nassays*2, ncells, nsamples))
        assert np.all(casv[:,:,0,0] == np.array([[1,1],
                                                 [0,1]]))
        assert np.all(casv[:,:,0,1] == np.array([[1,1],
                                                 [0,0]]))

    def test_compute_casv_radius(self):

        nregions = 3
        nassays = 1
        ncells = 1
        nsamples = 2
        radii = [1,3]
        indices = np.array([0,2])

        # m1: reference cells
        m1 = np.zeros((nregions, nassays, ncells))
        m1[0]=1 # first region has a signal
        m1[1]=1 # second region has a signal

        # m2: samples
        m2 = np.zeros((nregions, nassays, nsamples))
        m2[0,:,0] = 1 # first cell has 1 signal in region 1
        # second cell has 2 signals in first 2 regions
        m2[0:2,:,1] = 1

        casv = compute_casv(m1, m2, radii, indices = indices)
        assert(casv.shape == (len(indices), nassays*2 * len(radii), ncells, nsamples))

        # first region
        assert np.all(casv[0,:,0,0] == np.array([[1,0, 1,0.5]]))
        assert np.all(casv[0,:,0,1] == np.array([[1,0.5, 1,1]]))

        # second region
        assert np.all(casv[1,:,0,0] == np.array([[0, 0.5, 1, 0.5]]))
        assert np.all(casv[1,:,0,1] == np.array([[0,1,1,1]]))
