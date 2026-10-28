"""Independent analytic checks for space-time weak PDE regression."""
import unittest
import numpy as np
from weak_study import engine


class WeakBlockTests(unittest.TestCase):
    def setUp(self):
        self.x=np.arange(128)*2*np.pi/128
        self.t=np.linspace(0,1,101)

    def data(self,u,x=None,t=None):
        return {'x':self.x if x is None else x,'t':self.t if t is None else t,'u':u,'boundary':'periodic'}

    def test_advection_diffusion_signs_and_time_integration(self):
        c=.7; nu=.04
        u=sum(a*np.exp(-nu*k*k*self.t[:,None])*np.sin(k*(self.x[None,:]-c*self.t[:,None])+phase)
              for a,k,phase in [(1.,1,.3),(.4,2,-.1),(.2,3,.8)])
        for variant in engine.VARIANTS:
            b=engine.weak_block(self.data(u),variant)
            expected=-c*b['X'][:,4]+nu*b['X'][:,5]
            self.assertLess(np.linalg.norm(expected-b['y'])/np.linalg.norm(b['y']),2e-6)
            fitted=np.linalg.lstsq(b['X'][:,[4,5]],b['y'],rcond=1e-12)[0]
            np.testing.assert_allclose(fitted,[-c,nu],rtol=2e-6,atol=2e-7)

    def test_second_and_fourth_derivative_signs_on_scaled_domain(self):
        x=np.arange(128)*10*np.pi/128;t=np.linspace(0,2,101)
        u=sum(a*np.exp((k*k-k**4)*t[:,None])*np.cos(k*x[None,:]+phase)
              for a,k,phase in [(1.,.2,.1),(.3,.4,.5),(.2,.6,-.8)])
        for variant in engine.VARIANTS:
            b=engine.weak_block(self.data(u,x,t),variant)
            expected=-b['X'][:,5]-b['X'][:,7]
            self.assertLess(np.linalg.norm(expected-b['y'])/np.linalg.norm(b['y']),2e-6)

    def test_zero_mode_keeps_constant_source_and_polynomial_moments(self):
        u=np.broadcast_to((2+.3*self.t)[:,None],(101,128)).copy()
        b=engine.weak_block(self.data(u))
        zeros=b['X'][:,0]>.5
        self.assertGreater(int(zeros.sum()),1)
        np.testing.assert_allclose(b['y'][zeros],.3,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(b['X'][zeros,0],1,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(b['X'][:,4:],0,atol=1e-10)
        self.assertGreater(np.linalg.norm(b['X'][:,3]),np.linalg.norm(b['X'][:,2]))

    def test_cubic_and_conservative_transport_factors(self):
        # u=1+a*cos(x), constant in time. Fourier coefficients below are analytic.
        a=.6;u=np.broadcast_to(1+a*np.cos(self.x),(101,128)).copy()
        b=engine.weak_block(self.data(u))
        m=b['metadata']['max_spatial_mode']
        first=b['X'][:1+2*m]
        self.assertAlmostEqual(first[0,2],1+a*a/2,places=12)
        self.assertAlmostEqual(first[0,3],1+3*a*a/2,places=12)
        self.assertAlmostEqual(first[1,3]/np.sqrt(2),(3*a+3*a**3/4)/2,places=12)
        # 1/2*d_x(u^2) has complex +i*a/2 at Fourier mode +1.
        self.assertAlmostEqual(first[1+m,8]/np.sqrt(2),a/2,places=12)
        self.assertAlmostEqual(first[1+m,4]/np.sqrt(2),a/2,places=12)
        self.assertAlmostEqual(first[1+m,6]/np.sqrt(2),-a/2,places=12)
        np.testing.assert_allclose(b['y'],0,atol=1e-10)

    def test_alias_safe_retained_products_and_even_nyquist_split(self):
        n=64;x=np.arange(n)*2*np.pi/n
        u=np.broadcast_to(1+.3*np.cos(27*x)+.2*np.cos(32*x),(101,n)).copy()
        # 2N padding must not fold u^2's mode54 into retained mode10.
        b=engine.weak_block(self.data(u,x=x),variant='resolved')
        m=b['metadata']['max_spatial_mode'];first=b['X'][:1+2*m]
        self.assertAlmostEqual(first[0,2],1+.3**2/2+.2**2/2,places=11)
        self.assertAlmostEqual(first[10,2],0,places=11)
        self.assertAlmostEqual(first[5,2]/np.sqrt(2),.3*.2/2,places=11)

    def test_odd_grid_and_minimum_frames(self):
        x=np.arange(63)*2*np.pi/63;t=np.linspace(0,.1,16)
        u=np.exp(-.2*t[:,None])*np.cos(x)
        b=engine.weak_block(self.data(u,x,t))
        expected=.2*b['X'][:,5]
        np.testing.assert_allclose(b['y'],expected,rtol=1e-6,atol=1e-8)
        self.assertEqual(b['X'].shape[1],9)

    def test_only_supplied_observations_affect_features(self):
        rng=np.random.default_rng(871)
        data=self.data(np.sin(self.x)[None,:]+.01*rng.normal(size=(101,128)))
        a=engine.weak_block(data)
        data.update({'clean_u':np.zeros((101,128)),'sigma':999.,'reference_equation':'anything'})
        b=engine.weak_block(data)
        np.testing.assert_array_equal(a['X'],b['X']);np.testing.assert_array_equal(a['y'],b['y'])
        self.assertTrue(np.isfinite(a['X']).all());self.assertTrue(np.isfinite(a['y']).all())

    def test_temporal_quadrature_converges_on_fast_traveling_wave(self):
        errors=[]
        for count in (33,65,129):
            t=np.linspace(0,4,count)
            u=np.sin(self.x[None,:]-3*t[:,None])+.2*np.cos(2*(self.x[None,:]-3*t[:,None]))
            b=engine.weak_block(self.data(u,t=t))
            errors.append(np.linalg.norm(b['y']+3*b['X'][:,4])/np.linalg.norm(b['y']))
        self.assertLess(errors[1],errors[0]/100)
        self.assertLess(errors[2],errors[1]/100)
        self.assertLess(errors[2],1e-7)

    def test_rejects_invalid_fields_boundaries_and_variant(self):
        u=np.broadcast_to(np.sin(self.x),(101,128)).copy()
        for data in [None,{},dict(self.data(u),boundary='dirichlet'),self.data(u.astype(complex)),self.data(u[:5]),self.data(u,t=self.t[::-1])]:
            with self.assertRaises(ValueError):engine.weak_block(data)
        with self.assertRaises(ValueError):engine.weak_block(self.data(u),'unregistered')


if __name__=='__main__':unittest.main()
