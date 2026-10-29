"""Four-arm sharing and observation-only coefficient interfaces."""
from copy import deepcopy
from unittest.mock import patch
import unittest
import numpy as np
from coefficient_study import engine


def trajectory(marker):
    x=np.linspace(0,2*np.pi,32,endpoint=False);t=np.linspace(.05,.55,101)
    return {'x':x,'t':t,'u':np.tile(np.sin(x),(101,1))+marker,'boundary':'periodic',
        'prefix_t':np.linspace(0,.05,11),'prefix_u':np.tile(np.sin(x),(11,1))+marker}


def candidate():
    return {'candidate_id':'compact::fixture','weak_variant':'compact','terms':[{'name':'u_xx','coefficient':.1}],
        'support_size':1,'fit_success':True,'eligible':True,'development':{'rmse':1.},'equation':'u_t=0.1u_xx'}


class CoefficientEngineTests(unittest.TestCase):
    def test_fit_shares_two_pools_and_training_never_receives_prefix(self):
        train=[trajectory(.01)];dev=[trajectory(.02)];seen=[]
        def generate(values):
            self.assertEqual(set(values[0]),{'x','t','u','boundary'});seen.append(values)
            return [candidate()],[]
        def score(candidates,development):
            self.assertIs(development,dev)
            return {'selected':deepcopy(candidates[0]),'candidates':deepcopy(candidates),'near_optimal_supports':[['u_xx']],'failure_reason':None}
        def stable(candidates,training,development,block_builder):
            self.assertEqual(set(training[0]),{'x','t','u','boundary'});self.assertIs(development,dev)
            self.assertIn(block_builder,(engine.weak.weak_block,engine.noise.weak_block))
            return score(candidates,development)
        with patch.object(engine,'generate_candidates',side_effect=generate) as raw,patch.object(engine,'corrected_candidates',side_effect=generate) as noisy,patch.object(engine,'generate_support_medians',side_effect=lambda c:(c,{})),patch.object(engine.prefix,'select_candidates',side_effect=score) as scorer,patch.object(engine.selection,'select_candidates',side_effect=stable) as selector:
            results=engine.fit_observations(train,dev)
        self.assertEqual(set(results),set(engine.METHODS));self.assertEqual(scorer.call_count,2)
        self.assertEqual(selector.call_count,2);self.assertEqual(raw.call_count,1);self.assertEqual(noisy.call_count,1)
        self.assertTrue(results['noise_stable']['coefficient_noise_correction'])
        self.assertFalse(results['stable_selection']['coefficient_noise_correction'])

    def test_all_arms_keep_same_forecast_budget(self):
        for m in engine.METHODS:
            d=engine.stage_metadata(m)
            self.assertEqual(d['final_predictor'],'prefix_aligned_resolved_factor4')
            self.assertEqual(d['conditioning'],'observed_prefix')
        with self.assertRaises(ValueError):engine.stage_metadata('unknown')

    def test_corrected_fit_preserves_failed_variant_receipt(self):
        calls=[]
        def block(d,variant):
            calls.append(variant)
            if variant=='low':raise ValueError('manufactured noise failure')
            return {}
        with patch.object(engine.noise,'weak_block',side_effect=block),patch.object(engine.sparse,'fit_blocks',side_effect=lambda b,m:[candidate()]):
            candidates,trials=engine.corrected_candidates([trajectory(.01)])
        self.assertEqual(calls,list(engine.weak.VARIANTS));self.assertEqual(len(candidates),2)
        self.assertEqual(len(trials),3);self.assertIn('manufactured',trials[1]['failure_reason'])

    def test_empty_training_rejected(self):
        with self.assertRaises(ValueError):engine.fit_observations([],[])


if __name__=='__main__':unittest.main()
