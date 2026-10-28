"""Independent derivative checks for the reviewed Gaussian reference correction."""
import unittest

import numpy as np

from l1_study.gaussian_check import (
    SOURCE_COMMIT, gaussian_derivatives, incorrect_reference_third, verify,
)


class GaussianReferenceCorrectionTests(unittest.TestCase):
    def test_four_hundred_complex_step_checks_accept_fix_and_reject_original(self):
        result = verify()
        self.assertTrue(result["passed"])
        self.assertEqual(result["points_checked"], 400)
        self.assertEqual(result["source_commit"], SOURCE_COMMIT)
        for row in result["checks"]:
            self.assertLess(row["corrected_complex_step_max_abs_error"], 5e-13)
            self.assertGreater(row["original_complex_step_max_abs_error"], .1)

    def test_noncentral_hermite_zero_distinguishes_third_derivative(self):
        # H_3(z) has zeros at z=0 and z=+/-sqrt(3), while the erroneous
        # polynomial has noncentral zeros at +/-sqrt(5/3).
        sigma0, time = .7, .2
        variance = sigma0**2 + .1*time
        shifted = np.sqrt(3*variance)
        locations = .4*time + np.array([-shifted, 0., shifted])
        corrected = gaussian_derivatives(locations, time, sigma0)[2]
        np.testing.assert_allclose(corrected, 0, atol=2e-15)
        erroneous = incorrect_reference_third(locations, time, sigma0)
        self.assertGreater(abs(erroneous[0]), .1)
        self.assertGreater(abs(erroneous[2]), .1)


if __name__ == "__main__":
    unittest.main()
