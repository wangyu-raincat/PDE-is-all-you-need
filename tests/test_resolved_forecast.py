"""Independent manufactured checks for computational spatial refinement."""
import unittest
from unittest.mock import patch
import numpy as np

from resolved_study import forecast as f
from pde_discovery import generic_engine as old


class ResolvedForecastTests(unittest.TestCase):
    def test_factor_one_is_exactly_original_solver(self):
        x=np.arange(32)*2*np.pi/32;t=np.linspace(0,.1,6)
        u=.2+.3*np.sin(x)+.1*np.cos(13*x)
        terms=[{'name':'u_xx','coefficient':.05},{'name':'u*u_x','coefficient':-1.}]
        np.testing.assert_array_equal(f.forecast(terms,x,t,u,factor=1),old.forecast(terms,x,t,u))

    def test_high_mode_linear_transport_reaction_diffusion_exact(self):
        x=-.7+np.arange(32)*2*np.pi/32;t=np.linspace(2.,2.08,9);elapsed=t-t[0]
        u=.3+.4*np.sin(3*x)+.15*np.cos(13*x)
        terms=[{'name':'u','coefficient':.2},{'name':'u_x','coefficient':-.7},
               {'name':'u_xx','coefficient':.05},{'name':'u_xxxx','coefficient':-.0003}]
        expected=.3*np.exp(.2*elapsed)[:,None]+sum(
            amp*np.exp((.2-.05*m*m-.0003*m**4)*elapsed)[:,None]*trig(m*(x[None,:]-.7*elapsed[:,None]))
            for m,amp,trig in [(3,.4,np.sin),(13,.15,np.cos)])
        actual=f.forecast(terms,x,t,u)
        np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
        self.assertGreater(np.linalg.norm(old.forecast(terms,x,t,u)[0]-u),.1)

    def test_even_nyquist_and_odd_grid_real_interpolation(self):
        for n in (32,33):
            x=np.arange(n)*2*np.pi/n;t=np.linspace(0,.1,6);m=n//2
            u=.2+.3*np.cos(m*x)
            terms=[{'name':'u_x','coefficient':-.23},{'name':'u_xx','coefficient':.001}]
            expected=.2+.3*np.exp(-.001*m*m*t)[:,None]*np.cos(m*(x[None,:]-.23*t[:,None]))
            actual=f.forecast(terms,x,t,u)
            self.assertFalse(np.iscomplexobj(actual))
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)

    def test_viscous_burgers_matches_independent_cole_hopf_solution(self):
        n=32;x=np.arange(n)*2*np.pi/n;t=np.linspace(0,.5,26);nu=.1;a=.65
        amplitude=a*np.exp(-nu*t)
        expected=2*nu*amplitude[:,None]*np.sin(x)[None,:]/(1+amplitude[:,None]*np.cos(x)[None,:])
        terms=[{'name':'u_xx','coefficient':nu},{'name':'u*u_x','coefficient':-1.}]
        original=f.forecast(terms,x,t,expected[0],factor=1)
        twice=f.forecast(terms,x,t,expected[0],factor=2)
        refined=f.forecast(terms,x,t,expected[0],factor=4)
        self.assertLess(np.max(np.abs(refined-expected)),5e-6)
        self.assertLess(np.linalg.norm(refined[1:]-expected[1:]),.1*np.linalg.norm(original[1:]-expected[1:]))
        np.testing.assert_allclose(twice,refined,rtol=1e-7,atol=1e-8)

    def test_constant_forcing_uses_zero_mode_and_original_times(self):
        x=np.arange(17)*.1;t=np.array([1.,1.03,1.06,1.09]);u=np.full(17,.3)
        terms=[{'name':'1','coefficient':.2},{'name':'u','coefficient':-.1},
               {'name':'u*u_x','coefficient':-1.},{'name':'u_xx','coefficient':.01}]
        expected=2+(.3-2)*np.exp(-.1*(t-t[0]))
        result=f.forecast(terms,x,t,u)
        self.assertEqual(result.shape,(len(t),len(x)))
        np.testing.assert_allclose(result,np.broadcast_to(expected[:,None],result.shape),atol=2e-12,rtol=2e-12)

    def test_only_initial_data_accepted_and_inputs_unchanged(self):
        x=np.arange(32)/32;t=np.linspace(0,.05,5);u=.1+np.sin(2*np.pi*x)
        copies=[v.copy() for v in (x,t,u)]
        result=f.forecast([{'name':'u_xx','coefficient':.01}],x,t,u)
        np.testing.assert_allclose(result[0],u,atol=2e-14)
        for before,after in zip(copies,(x,t,u)):np.testing.assert_array_equal(before,after)
        with self.assertRaises(TypeError):f.forecast([],x,t,u,future=np.zeros((5,32)))
        with self.assertRaises(TypeError):f.forecast([],x,t,u,reference_terms=[])

    def test_invalid_complex_coordinates_states_and_factors_rejected(self):
        x=np.arange(32)/32;t=np.linspace(0,.1,4);u=np.sin(2*np.pi*x)
        for factor in (True,0,3,4.,8):
            with self.assertRaises(ValueError):f.forecast([],x,t,u,factor=factor)
        for xx,tt,uu in [(x+0j,t,u),(x,t+0j,u),(x,t,u+0j),(x,t,u[:-1]),(x,t,u*np.nan)]:
            with self.assertRaises(ValueError):f.forecast([],xx,tt,uu)

    def test_refinement_budget_rejected_without_silent_fallback(self):
        x=np.arange(513)/513;t=np.array([0.,1.]);u=np.zeros(513)
        with patch.object(old,'forecast',side_effect=AssertionError('solver must not run')) as solver:
            with self.assertRaises(ValueError):f.forecast([],x,t,u)
            solver.assert_not_called()

    def test_original_physical_and_time_budget_guards_remain(self):
        x=np.arange(32)/32;t=np.linspace(0,.1,4);u=np.sin(2*np.pi*x)
        for terms in [[{'name':'u_xx','coefficient':-.1}], [{'name':'u_xxxx','coefficient':.1}],
                      [{'name':'unknown','coefficient':1.}], [{'name':'u','coefficient':float('nan')}]]:
            with self.assertRaises(ValueError):f.forecast(terms,x,t,u)
        with self.assertRaises(ValueError):f.forecast([],x,np.linspace(0,1,1000),u)


if __name__=='__main__':unittest.main()
