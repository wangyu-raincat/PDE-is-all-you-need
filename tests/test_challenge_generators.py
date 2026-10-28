"""Independent reference checks; no formal training/development/final fields."""
import ast
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import resample

from challenge_study import generators as g


class ChallengeGeneratorTests(unittest.TestCase):
    def test_contour_zero_symbol_has_correct_runge_kutta_limits(self):
        step=.013
        e,e2,q,f1,f2,f3=g._contour_coefficients(np.zeros(4),step)
        np.testing.assert_allclose(e,1,atol=1e-14)
        np.testing.assert_allclose(e2,1,atol=1e-14)
        for value,expected in [(q,step/2),(f1,step/6),(f2,step/6),(f3,step/6)]:
            np.testing.assert_allclose(value,expected,rtol=2e-14,atol=2e-16)

    def test_linear_fourier_reference_is_exact_at_irregular_output_times(self):
        x=np.linspace(0,2*np.pi,32,endpoint=False)
        t=np.array([3.0,3.011,3.045,3.083])
        initial=.1+.8*np.sin(x)+.2*np.cos(3*x)
        expected=.1+.8*np.exp((-.12-.003)*(t-t[0]))[:,None]*np.sin(x) + .2*np.exp((-.12*9-.003*81)*(t-t[0]))[:,None]*np.cos(3*x)
        actual=g._integrate(initial,2*np.pi,t,{2:.12,4:-.003},0,.006)
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)

    def test_nonlinear_contour_solution_matches_independent_dop853(self):
        n=32;period=2*np.pi;x=np.arange(n)*period/n
        initial=.1+.7*np.sin(x+.2)+.2*np.cos(2*x-.3)+.1*np.sin(3*x)
        wave=np.fft.fftfreq(n)*n;kept=np.abs(wave)<=n//3
        t=np.linspace(0,.16,17)
        for linear in ({2:.03},{2:-.06,4:-.004}):
            def rhs(_t,u):
                spectrum=np.fft.fft(u)*kept
                ux=np.fft.ifft(1j*wave*spectrum).real
                # Independent 3N interpolation and u*ux, not conservative u².
                nonlinear=resample(-resample(u,3*n)*resample(ux,3*n),n)
                nonlinear=np.fft.ifft(np.fft.fft(nonlinear)*kept).real
                result=nonlinear
                for order,value in linear.items():
                    result=result+value*np.fft.ifft((1j*wave)**order*spectrum).real
                return result
            reference=solve_ivp(rhs,(t[0],t[-1]),initial,t_eval=t,method="DOP853",
                rtol=2e-12,atol=2e-13,max_step=.0005)
            self.assertTrue(reference.success)
            actual=g._integrate(initial,period,t,linear,-1,.0005)
            np.testing.assert_allclose(actual,reference.y.T,rtol=2e-9,atol=2e-10)

    def test_role_seeds_are_disjoint_without_generating_role_fields(self):
        seeds=[g.ROLE_SEEDS[role]+i for role,count in g.ROLE_COUNTS.items() for i in range(count)]
        self.assertEqual(len(seeds),len(set(seeds)))
        for family in g.FAMILIES:
            first=g._initial_parameters(family,"calibration",0)
            second=g._initial_parameters(family,"calibration",0)
            np.testing.assert_array_equal(first[2],second[2])
            np.testing.assert_array_equal(first[3],second[3])

    def test_invalid_roles_indices_and_numeric_inputs_rejected(self):
        for args in [("bad","calibration",0),(g.FAMILIES[0],"bad",0),
                     (g.FAMILIES[0],"calibration",1),(g.FAMILIES[0],"calibration",True)]:
            with self.assertRaises(ValueError):g.generate(*args)
        with self.assertRaises(ValueError):g.generate(g.FAMILIES[0],"calibration",refined=1)
        for initial,t,linear,dt in [(np.ones(15),[0,1],{2:.1},.01),
                (np.ones(16),[0,0],{2:.1},.01),(np.ones(16),[0,1],{3:1},.01),
                (np.ones(16),[0,1],{2:.1},0)]:
            with self.assertRaises(ValueError):g._integrate(initial,2*np.pi,t,linear,-1,dt)

    def test_generator_cannot_import_candidate_or_forecast_implementation(self):
        source=inspect.getsource(g)
        tree=ast.parse(source)
        imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
            elif isinstance(node,ast.ImportFrom):imports.append(node.module or "")
        self.assertFalse(any(name.startswith(("pde_discovery","l1_study")) for name in imports))
        generate_tree=ast.parse(inspect.getsource(g.generate))
        calls=[n.func.id for n in ast.walk(generate_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertNotIn("reference_terms",calls)

    def test_calibration_uses_only_reserved_roles_and_keeps_failure(self):
        calls=[]
        def fixture(family,role,index,refined):
            calls.append((family,role,index,refined))
            u=np.ones((3,4))
            if family==g.FAMILIES[1] and refined:u[1:]+=0.01
            return {"x":np.arange(4),"t":np.arange(3),"u":u,"metadata":{"refined":refined}}
        with patch.object(g,"generate",side_effect=fixture):result=g.calibration()
        self.assertFalse(result["all_passed"])
        self.assertEqual(result["records"][0]["status"],"passed")
        self.assertEqual(result["records"][1]["status"],"source_quality_failure")
        self.assertEqual(len(calls),4)
        self.assertTrue(all(role=="calibration" and index==0 for _,role,index,_ in calls))


if __name__ == "__main__":
    unittest.main()
