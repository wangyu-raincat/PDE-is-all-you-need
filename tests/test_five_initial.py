"""Independent signal invariances and adverse-input checks for initial filtering."""
import unittest
import numpy as np
from five_study.initial import denoise_initial


class InitialDenoisingTests(unittest.TestCase):
    def grid(self, n=128): return np.arange(n)*2*np.pi/n

    def test_noiseless_sparse_low_and_high_modes_with_phases_are_preserved(self):
        for n, modes in [(16,(1,7)), (32,(3,14)), (127,(4,57)), (128,(3,53))]:
            x=self.grid(n)
            u=.4+.7*np.sin(modes[0]*x+.27)+.3*np.cos(modes[1]*x-.61)
            filtered, d=denoise_initial(x,u)
            np.testing.assert_allclose(filtered,u,rtol=2e-13,atol=2e-13)
            self.assertTrue(set(modes).issubset(d['retained_positive_modes']))

    def test_even_nyquist_and_odd_highest_mode_remain_real_and_present(self):
        for n in (32,33):
            x=self.grid(n);m=n//2
            u=.2+.3*np.cos(m*x)
            filtered,d=denoise_initial(x,u)
            self.assertFalse(np.iscomplexobj(filtered))
            np.testing.assert_allclose(filtered,u,atol=2e-14)
            self.assertIn(m,d['retained_positive_modes'])
            self.assertLess(d['scale_band_stop_exclusive'],m+1)

    def test_constants_and_zero_preserve_mean_without_invalid_diagnostics(self):
        for n in (16,31,128):
            for value in (0.,.37,-4.):
                u=np.full(n,value);filtered,d=denoise_initial(self.grid(n),u)
                np.testing.assert_allclose(filtered,u,atol=2e-14)
                self.assertAlmostEqual(filtered.mean(),value,places=13)
                for key in ('estimated_spatial_noise_sigma','relative_initial_change',
                            'removed_non_dc_energy_fraction','coefficient_threshold'):
                    self.assertTrue(np.isfinite(d[key]))

    def test_white_noise_scale_calibration_does_not_confuse_complex_variance(self):
        # Independent repeated noise samples test the physical scale, rather
        # than duplicating the implementation's formula as the expected value.
        rng=np.random.default_rng(2034);sigma=.07;x=self.grid(256)
        estimated=[]
        for _ in range(200):
            _,d=denoise_initial(x,rng.normal(0,sigma,len(x)))
            estimated.append(d['estimated_spatial_noise_sigma'])
        self.assertLess(abs(float(np.mean(estimated))/sigma-1),.05)

    def test_sparse_signal_denoising_reduces_noise_and_keeps_high_frequency_signal(self):
        rng=np.random.default_rng(345);x=self.grid(128)
        truth=.2+.5*np.sin(3*x+.2)+.24*np.cos(47*x-.4)
        raw=truth+rng.normal(0,.02,len(x))
        filtered,d=denoise_initial(x,raw)
        self.assertLess(np.linalg.norm(filtered-truth),.5*np.linalg.norm(raw-truth))
        self.assertIn(47,d['retained_positive_modes'])
        self.assertGreater(d['removed_positive_mode_count'],20)
        self.assertAlmostEqual(filtered.mean(),raw.mean(),places=14)

    def test_circular_translation_equivariance_and_retained_phase(self):
        x=self.grid(128);rng=np.random.default_rng(784)
        u=.3+np.cos(5*x+.3)+rng.normal(0,.03,len(x))
        filtered,d=denoise_initial(x,u)
        shifted,ds=denoise_initial(x,np.roll(u,19))
        np.testing.assert_allclose(shifted,np.roll(filtered,19),atol=2e-14)
        self.assertEqual(ds['retained_positive_modes'],d['retained_positive_modes'])
        self.assertAlmostEqual(ds['estimated_spatial_noise_sigma'],d['estimated_spatial_noise_sigma'],places=14)
        before=np.fft.rfft(u);after=np.fft.rfft(filtered)
        np.testing.assert_allclose(after[d['retained_positive_modes']],before[d['retained_positive_modes']],atol=2e-14)

    def test_field_units_sign_offset_and_spatial_units_equivariance(self):
        x=self.grid(128);rng=np.random.default_rng(187)
        u=np.sin(4*x+.6)+rng.normal(0,.01,len(x))
        filtered,d=denoise_initial(x,u)
        changed,dc=denoise_initial(3+100*x,-2.5*u+7.)
        np.testing.assert_allclose(changed,-2.5*filtered+7.,atol=3e-14)
        self.assertEqual(dc['retained_positive_modes'],d['retained_positive_modes'])
        self.assertAlmostEqual(dc['estimated_spatial_noise_sigma'],2.5*d['estimated_spatial_noise_sigma'],places=13)

    def test_no_input_mutation_and_no_future_or_oracle_arguments(self):
        x=self.grid(32);u=np.sin(x);before=(x.copy(),u.copy())
        filtered,d=denoise_initial(x,u)
        np.testing.assert_array_equal(x,before[0]);np.testing.assert_array_equal(u,before[1])
        self.assertFalse(np.shares_memory(filtered,u))
        for extra in ('future','family','noise_sigma','reference_terms'):
            with self.assertRaises(TypeError):denoise_initial(x,u,**{extra:0})
        self.assertFalse(d['reference_used']);self.assertFalse(d['future_used'])

    def test_nonuniform_complex_nonfinite_and_budget_inputs_fail_closed(self):
        x=self.grid(32);u=np.sin(x)
        cases=[(x.astype(complex),u),(x,u.astype(complex)),(x,u[:-1]),
               (x,np.full(32,np.nan)),(x,np.full(32,1e7)),(x[::-1],u),
               (np.r_[x[:-1],x[-1]+.01],u),(x[:,None],u),
               (self.grid(15),np.ones(15)),(self.grid(513),np.ones(513))]
        for xx,uu in cases:
            with self.subTest(shape=np.shape(xx)):
                with self.assertRaises(ValueError):denoise_initial(xx,uu)

    def test_dense_high_frequency_signal_limitation_is_exposed(self):
        # This intentionally adverse noiseless signal occupies the estimation
        # band. Its change must be reported, not advertised as noise recovery.
        x=self.grid(64)
        u=np.sum([.05*np.cos(k*x+.17*k) for k in range(16,32)],axis=0)
        filtered,d=denoise_initial(x,u)
        self.assertGreater(np.linalg.norm(filtered-u),.1)
        self.assertGreater(d['relative_initial_change'],.1)
        self.assertTrue(d['broadband_signal_may_be_removed'])


if __name__=='__main__':unittest.main()
