from utils import *
from unittest import TestCase

"""
- For each operation, you should write tests to test  on matrices of different sizes.
- Keep in mind that the tests provided in the starter code are NOT comprehensive. That is, we strongly
advise you to modify them and add new tests.
- Hint: use dp_mc_matrix to generate dumbpy and numc matrices with the same data and use
      cmp_dp_nc_matrix to compare the results
"""
class TestAdd(TestCase):
    def test_small_add(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_add(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(255, 20, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(255, 20, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_add(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(12000, 10000, seed=2)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(12000, 10000, seed=5)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "add")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestSub(TestCase):
    def test_small_sub(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(2, 2, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(2, 2, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        try:
            nc.Matrix(3, 3) - nc.Matrix(2, 2)
            self.assertTrue(False)
        except ValueError as e:
            print(e)
            pass
        print_speedup(speed_up)

    def test_medium_sub(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(288, 299, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(288, 299, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_sub(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(12000, 12000, seed=1)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(12000, 12000, seed=3)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "sub")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestAbs(TestCase):
    def test_small_abs(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_abs(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_abs(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(12000, 12000, seed=4)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "abs")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestNeg(TestCase):
    def test_small_neg(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
    def test_medium_neg(self):
        # TODO: YOUR CODE HERE
        pass

    def test_large_neg(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(12000, 12000, seed=0)
        is_correct, speed_up = compute([dp_mat], [nc_mat], "neg")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestMul(TestCase):
    def test_small_mul(self):
        # TODO: YOUR CODE HERE
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(259, 321, seed=0)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(321, 283, seed=1)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_mul(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1000, 1300, seed=5)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1300, 1000, seed=6)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)
        


    def test_large_mul(self):
        dp_mat1, nc_mat1 = rand_dp_nc_matrix(1200, 1203, seed=5)
        dp_mat2, nc_mat2 = rand_dp_nc_matrix(1203, 1200, seed=0)
        is_correct, speed_up = compute([dp_mat1, dp_mat2], [nc_mat1, nc_mat2], "mul")
        self.assertTrue(is_correct)
        print_speedup(speed_up)


class TestPow(TestCase):
    def test_small_pow(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        is_correct, speed_up = compute([dp_mat, 0], [nc_mat, 0], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_medium_pow(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(120, 120, seed=3)
        is_correct, speed_up = compute([dp_mat, 103], [nc_mat, 103], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

    def test_large_pow(self):
        dp_mat, nc_mat = rand_dp_nc_matrix(300, 300, seed=3)
        is_correct, speed_up = compute([dp_mat, 500], [nc_mat, 500], "pow")
        self.assertTrue(is_correct)
        print_speedup(speed_up)

class TestGet(TestCase):
    def test_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat.get(rand_row, rand_col), decimal_places),
            round(nc_mat.get(rand_row, rand_col), decimal_places))

class TestSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        dp_mat.set(rand_row, rand_col, 2)
        nc_mat.set(rand_row, rand_col, 2)
        self.assertTrue(cmp_dp_nc_matrix(dp_mat, nc_mat))
        self.assertEqual(nc_mat.get(rand_row, rand_col), 2)

class TestShape(TestCase):
    def test_shape(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(dp_mat.shape == nc_mat.shape)

class TestIndexGet(TestCase):
    def test_index_get(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        self.assertEqual(round(dp_mat[rand_row][rand_col], decimal_places),
            round(nc_mat[rand_row][rand_col], decimal_places))

class TestIndexSet(TestCase):
    def test_set(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        rand_row = np.random.randint(dp_mat.shape[0])
        rand_col = np.random.randint(dp_mat.shape[1])
        dp_mat[rand_row][rand_col] = 2
        nc_mat[rand_row][rand_col] = 2
        self.assertTrue(cmp_dp_nc_matrix(dp_mat, nc_mat))
        self.assertEquals(nc_mat[rand_row][rand_col], 2)

class TestSlice(TestCase):
    def test_slice(self):
        # TODO: YOUR CODE HERE
        dp_mat, nc_mat = rand_dp_nc_matrix(2, 2, seed=0)
        self.assertTrue(cmp_dp_nc_matrix(dp_mat[0], nc_mat[0]))
        self.assertTrue(cmp_dp_nc_matrix(dp_mat[1], nc_mat[1]))
