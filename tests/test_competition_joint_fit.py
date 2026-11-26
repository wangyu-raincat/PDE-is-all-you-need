import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from pde_discovery.real_heat_refined import _prepare, _advance
from pde_discovery.heat_identifiability import fit_masked
from competition_agent.scenario_am import joint_fit as j


class JointFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol,_ = j.frozen_protocol()

    def records(self):
        rows=[]
        for i,n in enumerate((41,61)):
            t=np.linspace(0,120,n)
            bounds=np.column_stack((20+3*np.sin(t/30),5+np.cos(t/20)))
            u0=np.array([16,12,9,7.])
            interior=_advance(_prepare(t,bounds,u0,.007),5e-6,.01+.005*i,.02-.01*i)
            rows.append(dict(id=str(i),t=t,u=np.column_stack((bounds[:,0],interior,bounds[:,1])),dx=.007))
        return rows

    def test_joint_recovers_shared_D_and_separate_nuisance(self):
        fit=j.fit_records(self.records(),self.protocol)
        self.assertTrue(fit['success'])
        self.assertAlmostEqual(fit['D']/5e-6,1,places=5)
        self.assertLess(fit['rmse_K'],1e-6)
        self.assertAlmostEqual(fit['parameters']['1']['h'],.015,places=6)
        self.assertEqual(len(fit['starts']),3)

    def test_equal_record_weight_and_original_single_method(self):
        records=self.records();fixed=8e-6
        fit=j.fit_records(records,self.protocol,fixed_D=fixed)
        errors=[]
        for r in records:
            hc=fit['parameters'][r['id']]
            pred=_advance(_prepare(r['t'],r['u'][:,[0,5]],r['u'][0,1:5],r['dx']),fixed,hc['h'],hc['c'])
            errors.append(np.mean((pred[1:]-r['u'][1:,1:5])**2))
        self.assertAlmostEqual(fit['rmse_K'],np.sqrt(np.mean(errors)),places=12)
        r=records[0]
        own=j.fit_records([r],self.protocol,fixed_D=fixed)
        old=fit_masked(r['t'],r['u'],r['dx'],'D+h+c',fixed_D=fixed)
        self.assertAlmostEqual(own['D'],old['parameters']['D'])
        self.assertAlmostEqual(own['parameters']['0']['h'],old['parameters']['h'],places=7)

    def test_exact_thresholds_and_missing_fold(self):
        fit={'success':True};profile={'complete':True,'ratio':1.5}
        folds=[{'success':True,'D':v} for v in (1,1.3,1.1,1.2,1)]
        self.assertTrue(j.rod_decision(fit,profile,folds,self.protocol)['identifiable'])
        self.assertFalse(j.rod_decision(fit,dict(profile,ratio=1.50000001),folds,self.protocol)['identifiable'])
        self.assertFalse(j.rod_decision(fit,profile,folds[:4],self.protocol)['identifiable'])

    def test_cross_rod_requires_both_and_strict_ratio(self):
        rods={g:dict(joint={'D':d},decision={'identifiable':True,'loro_ratio':r})
              for g,d,r in [('xp',1.5,1.5),('xn',1,1)]}
        self.assertFalse(j.compare_rods(rods)['distinguishable'])
        rods['xp']['joint']['D']=1.50001
        self.assertTrue(j.compare_rods(rods)['distinguishable'])
        rods['xn']['decision']['identifiable']=False
        self.assertFalse(j.compare_rods(rods)['distinguishable'])

    def test_profile_retains_clipped_grid_and_failures(self):
        def fake(records,protocol,fixed_D=None):
            return dict(success=fixed_D!=4e-8,D=fixed_D,rmse_K=1.)
        with patch.object(j,'fit_records',side_effect=fake):
            result=j.profile_records([],{'success':True,'D':1e-8},self.protocol)
        self.assertEqual(len(result['mapping']),9)
        self.assertTrue(result['mapping'][0]['clipped'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['touches_grid_edge'])

    def test_protocol_guard_runs_before_data(self):
        with patch.object(j,'frozen_protocol',side_effect=ValueError('changed')),patch.object(j,'load_measurements') as load:
            with self.assertRaises(ValueError):j.heat_joint_fit('xn')
            load.assert_not_called()

    def test_git_guard_does_not_inherit_mcp_input(self):
        with patch.object(j.subprocess,'check_output',return_value=j.PROTOCOL.read_bytes()) as show, \
             patch.object(j.subprocess,'run') as ancestor:
            j.frozen_protocol()
        for call in (show.call_args,ancestor.call_args):
            self.assertEqual(call.kwargs['stdin'],j.subprocess.DEVNULL)
            self.assertEqual(call.kwargs['stderr'],j.subprocess.PIPE)
            self.assertEqual(call.kwargs['timeout'],30)

    def test_loro_omits_entire_record_from_fit(self):
        record=self.records()[0]
        calls=[]
        def fake(records,protocol,fixed_D=None):
            calls.append([r['id'] for r in records])
            return {'success':True,'D':1e-5}
        with patch.object(j,'load_measurements',return_value=(record['t'],record['u'])), \
             patch.object(j,'fit_records',side_effect=fake), \
             patch.object(j,'profile_records',return_value={'complete':True,'ratio':1.0}):
            result=j.heat_joint_fit('xn')
        folds=[ids for ids in calls if len(ids)==4]
        self.assertEqual(len(folds),5)
        for i,ids in enumerate(folds,1):
            self.assertEqual(set(ids),{f'xn{k}' for k in range(1,6) if k!=i})
        self.assertEqual([r['omitted'] for r in result['loro']],[f'xn{i}' for i in range(1,6)])


if __name__=='__main__':unittest.main()
