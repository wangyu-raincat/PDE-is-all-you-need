"""Support-safe training aggregation and consistently filtered forecast selection."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits

from five_study import engine as f
from five_study import forecast as predictor


def candidate(identity, variant, coefficient=.1, extra=None):
    terms = [{'name': 'u_xx', 'coefficient': coefficient}]
    if extra is not None:
        terms.append({'name': 'u', 'coefficient': extra})
    return {'candidate_id': identity, 'weak_variant': variant, 'terms': terms,
            'support_size': len(terms), 'fit_success': True,
            'alpha_values': [.001], 'diagnostics': {'training_marker': identity}}


def field(second=False):
    x = np.arange(32)*2*np.pi/32
    t = np.linspace(0, .2, 33)
    u = np.exp(-.1*t[:, None])*np.cos(x)[None, :]
    u += .2*np.exp(-.4*t[:, None])*np.sin(2*x)[None, :]
    if second:
        u += .05*np.exp(-.9*t[:, None])*np.cos(3*x)[None, :]
    return {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}


class FiveEngineTests(unittest.TestCase):
    def setUp(self):
        self.members = [candidate('compact-a','compact',.1),
                        candidate('low-a','low',.2),
                        candidate('resolved-a','resolved',.9)]

    def test_all_supports_receive_separate_training_only_candidates(self):
        members = self.members + [candidate('compact-b','compact',.03,-1.),
                                  candidate('low-b','low',.05,-3.)]
        pool, evidence = f.generate_support_medians(members)
        medians = pool[len(members):]
        self.assertEqual(len(medians),2)
        by_size={c['support_size']:c for c in medians}
        self.assertEqual(by_size[1]['terms'],[{'name':'u_xx','coefficient':.2}])
        self.assertEqual(by_size[2]['terms'],[{'name':'u','coefficient':-2.},
                                              {'name':'u_xx','coefficient':.04}])
        self.assertTrue(evidence['training_only'])
        self.assertFalse(evidence['development_used'])
        self.assertEqual(pool[:len(members)],members)

    def test_exact_variant_aliases_do_not_double_weight_coefficients(self):
        members=[candidate('compact','compact',.1),candidate('resolved','resolved',.5)]
        base,evidence=f.generate_support_medians(members)
        with_alias,alias_evidence=f.generate_support_medians(members+[candidate('low','low',.1)])
        self.assertEqual(base[-1]['terms'],with_alias[-1]['terms'])
        self.assertEqual(with_alias[-1]['terms'][0]['coefficient'],.3)
        aliases=alias_evidence['groups'][0]['distinct_vector_groups']
        self.assertEqual(len(aliases),2)
        self.assertEqual(aliases[0]['variants'],['compact','low'])
        self.assertEqual(aliases[0]['candidate_ids'],['compact','low'])

    def test_identical_vectors_produce_no_duplicate_median(self):
        members=[candidate(v,v,.1) for v in ('compact','low','resolved')]
        pool,evidence=f.generate_support_medians(members)
        self.assertEqual(len(pool),3)
        self.assertEqual(evidence['added_candidate_count'],0)
        self.assertEqual(evidence['groups'][0]['distinct_vector_count'],1)

    def test_path_multiplicity_and_variant_duplicates_do_not_vote(self):
        duplicates=[dict(candidate('compact-z'+str(i),'compact',7.),alpha_values=[.1]*500)
                    for i in range(8)]
        pool,evidence=f.generate_support_medians(self.members+duplicates)
        self.assertEqual(pool[-1]['terms'][0]['coefficient'],.2)
        self.assertEqual(evidence['groups'][0]['variant_representative_ids'],
                         ['compact-a','low-a','resolved-a'])

    def test_saved_development_scores_and_eligibility_do_not_affect_training_median(self):
        mutated=deepcopy(self.members)
        for i,c in enumerate(mutated):
            c['eligible']=False
            c['development']={'rmse':[float('nan'),1e8,0.][i]}
        pool,evidence=f.generate_support_medians(mutated)
        self.assertEqual(pool[-1]['terms'],[{'name':'u_xx','coefficient':.2}])
        self.assertEqual(evidence['added_candidate_count'],1)

    def test_failed_unknown_or_malformed_fit_cannot_contribute(self):
        malformed=deepcopy(self.members)
        malformed[0]['fit_success']=False
        malformed[1]['weak_variant']='oracle'
        malformed[2]['terms'][0]['coefficient']=float('nan')
        pool,evidence=f.generate_support_medians(malformed)
        self.assertEqual(len(pool),len(malformed))
        self.assertEqual(evidence['added_candidate_count'],0)
        self.assertEqual(len(evidence['excluded_candidates']),3)

    def test_zero_median_is_rejected_without_deleting_terms_or_originals(self):
        members=[candidate('a','compact',-.2),candidate('b','low',.2)]
        pool,evidence=f.generate_support_medians(members)
        self.assertEqual(pool,members)
        self.assertEqual(evidence['groups'][0]['status'],'invalid_median')
        self.assertEqual(evidence['groups'][0]['proposed_terms'],[{'name':'u_xx','coefficient':0.}])

    def test_overflow_median_records_json_safe_failure(self):
        members=[candidate('a','compact',1e308),candidate('b','low',1.6e308)]
        pool,evidence=f.generate_support_medians(members)
        self.assertEqual(pool,members)
        self.assertEqual(evidence['groups'][0]['status'],'invalid_median')
        self.assertIsNone(evidence['groups'][0]['proposed_terms'][0]['coefficient'])

    def test_inputs_are_not_mutated_by_augmentation_or_scored_choice(self):
        before=deepcopy(self.members)
        pool,evidence=f.generate_support_medians(self.members)
        pool[0]['terms'][0]['coefficient']=22.
        evidence['groups'][0]['distinct_vector_groups'][0]['coefficients'][0]=33.
        self.assertEqual(self.members,before)
        scored=[dict(c,eligible=True,development={'rmse':float(i+1)}) for i,c in enumerate(self.members)]
        saved=deepcopy(scored);result=f.choose_scored(scored)
        result['selected']['terms'][0]['coefficient']=44.
        self.assertEqual(scored,saved)

    def test_near_optimal_selection_can_change_structure_without_old_anchor(self):
        dense=candidate('dense','compact',.1,-1.)
        sparse=candidate('sparse','low',.2)
        worse=candidate('too-far','resolved',.9)
        scored=[dict(c,eligible=True,development={'rmse':e})
                for c,e in [(dense,1.),(sparse,1.04),(worse,1.06)]]
        result=f.choose_scored(scored)
        self.assertEqual(result['selected']['candidate_id'],'sparse')
        self.assertEqual(len(result['near_optimal_supports']),2)

    def test_scored_choice_rejects_nonfinite_or_mismatched_support_metadata(self):
        scored=[dict(c,eligible=True,development={'rmse':1.}) for c in self.members]
        scored[0]['development']['rmse']=float('nan')
        scored[1]['support_size']=2
        scored[2]['fit_success']=False
        result=f.choose_scored(scored)
        self.assertIsNone(result['selected'])
        self.assertTrue(all(not c['eligible'] for c in result['candidates']))

    def test_selector_initial_only_scoring_and_failed_candidate_stale_score_removed(self):
        dev=field();members=deepcopy(self.members)
        members[2]['fit_success']=False
        members[2]['eligible']=True;members[2]['development']={'rmse':0.}
        def forecast(terms,x,t,initial):
            np.testing.assert_array_equal(initial,dev['u'][0])
            self.assertEqual(initial.ndim,1)
            prediction=dev['u'].copy();prediction[1:]+=terms[0]['coefficient']
            return prediction,{'future_used':False,'estimated_sigma':.01}
        with patch.object(f,'forecast_with_diagnostics',side_effect=forecast) as mock:
            result=f.select_candidates(members,[dev])
        self.assertEqual(mock.call_count,2)
        self.assertEqual(result['selected']['candidate_id'],'compact-a')
        self.assertNotIn('development',result['candidates'][2])
        self.assertTrue(result['selected']['development']['initial_filter_diagnostics'])

    def test_forecast_failures_retained_without_fake_eligible_candidate(self):
        with patch.object(f,'forecast_with_diagnostics',side_effect=RuntimeError('manufactured failure')):
            result=f.select_candidates(self.members,[field()])
        self.assertIsNone(result['selected'])
        self.assertEqual(len(result['candidates']),3)
        self.assertTrue(all(c['rejection_reason']=='manufactured failure' for c in result['candidates']))

    def test_forecast_wrapper_filters_only_initial_and_returns_same_diagnostics(self):
        d=field();terms=[{'name':'u_xx','coefficient':.1}]
        filtered=d['u'][0]+.01;diagnostics={'sentinel':'single-frame'}
        def advance(actual_terms,x,t,u0):
            self.assertIs(actual_terms,terms)
            np.testing.assert_array_equal(u0,filtered)
            return np.tile(u0,(len(t),1))
        with patch.object(predictor,'denoise_initial',return_value=(filtered,diagnostics)) as denoise, \
             patch.object(predictor.refined,'forecast',side_effect=advance):
            predicted,record=predictor.forecast_with_diagnostics(terms,d['x'],d['t'],d['u'][0])
        self.assertEqual(denoise.call_count,1)
        self.assertIs(record,diagnostics)
        np.testing.assert_array_equal(predicted[0],filtered)

    def test_wrapper_scores_augmented_pool_once_and_preserves_raw_baselines(self):
        def score(candidates,dev):
            scored=[dict(deepcopy(c),eligible=True,development={'rmse':1.}) for c in candidates]
            return f.choose_scored(scored)
        original=score(self.members,[]);resolved=score(self.members,[])
        old_fused=deepcopy(resolved);old_fused['fusion']={'unchanged':True}
        with patch.object(f.previous,'generate_candidates',return_value=(self.members,[])) as generate, \
             patch.object(f.sparse,'select_candidates',return_value=deepcopy(original)) as old_score, \
             patch.object(f.refined,'select_candidates',return_value=deepcopy(resolved)) as raw_refined, \
             patch.object(f.previous,'select_from_scored',return_value=deepcopy(old_fused)) as old_fuse, \
             patch.object(f,'select_candidates',side_effect=score) as new_score:
            results=f.fit_observations([field()],[field()])
        generate.assert_called_once();old_score.assert_called_once();raw_refined.assert_called_once()
        old_fuse.assert_called_once();new_score.assert_called_once()
        self.assertEqual(set(results),set(f.METHODS))
        self.assertEqual(len(results['denoised_weak']['candidates']),3)
        self.assertEqual(len(results['denoised_median']['candidates']),4)
        self.assertEqual(results['weak_resolved']['selected'],resolved['selected'])
        self.assertEqual(results['fused_median']['fusion'],{'unchanged':True})
        for name in ('denoised_weak','denoised_median'):
            self.assertEqual(results[name]['initial_filter'],'spectral_hard')
            self.assertEqual(results[name]['structural_selection_predictor'],'denoised_resolved_factor4')
        self.assertEqual(results['fused_median']['structural_selection_predictor'],'original_observed_grid')
        self.assertEqual(results['fused_median']['initial_filter'],'none')

    def test_real_manufactured_data_raw_controls_match_historical_methods(self):
        with threadpool_limits(limits=1):
            observed=[field()];dev=[field(True)]
            previous=f.previous.fit_observations(observed,dev)
            actual=f.fit_observations(observed,dev)
        for method in ('weak_resolved','fused_median'):
            stripped={k:v for k,v in actual[method].items() if k not in ('initial_filter','median_augmentation')}
            self.assertEqual(stripped,previous[method])
        for method in ('denoised_weak','denoised_median'):
            self.assertIsNotNone(actual[method]['selected'])
            self.assertEqual(actual[method]['selected']['development']['trajectory_count'],1)

    def test_no_reference_family_noise_or_future_interfaces(self):
        with self.assertRaises(TypeError):f.fit_observations([],[],family='allen_cahn')
        with self.assertRaises(TypeError):f.generate_support_medians([],reference=[])
        with self.assertRaises(TypeError):predictor.forecast([],[],[],[],true_noise=.1)
        with self.assertRaises(ValueError):f.select_candidates([],[])


if __name__=='__main__':
    unittest.main()
