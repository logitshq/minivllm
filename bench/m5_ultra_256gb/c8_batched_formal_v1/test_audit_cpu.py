"""Three tiny E055 reducer contracts; no model, actual prepare or GPU."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_e055_test_reducer',HERE/'audit.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
PHASES=('r1-c8-on','r1-c8-off','r2-c8-off','r2-c8-on','r3-c8-on','r3-c8-off')

def fixture():
    points=[];ids={}
    for label in PHASES:
        repeat=int(label[1]);arm=label.rsplit('-',1)[1];on=arm=='on'
        points.append(dict(phase=label,repeat=repeat,concurrency=8,arm=arm,requests=64,
            aggregate_output_tok_s=110. if on else 100.,e2e_p50_s=9. if on else 10.,e2e_p95_s=10. if on else 11.))
        ids[label]={phase:[dict(fixture_id='D'+str(i%8),occurrence=i//8,raw_ids=[7,8],committed_ids=[7,8],terminal_eos_id=None)
                           for i in range(n)] for phase,n in (('warm',8),('formal',64))}
    return points,ids


class ReducerTests(unittest.TestCase):
    def test_three_pairs_and_actual_preregistered_identity_match(self):
        points,ids=fixture();result=m.reduce_pairs(points,ids,PHASES,m.DECISION)
        self.assertEqual(len(result['pairs']),3);self.assertTrue(result['speed_gate_passed'])
        self.assertTrue(result['all_pair_output_ids_identical']);self.assertAlmostEqual(result['median_pair_ratios']['throughput_ratio'],1.1)
        self.assertEqual(result['off_rate_spread_fraction'],0)
        # Closed preregistration metadata only, not live outputs or actual prepare.
        trial=m.load(m.TRIAL,m.TRIAL_SHA);server=trial.module(trial.SERVER);formal=trial.module(trial.FORMAL)
        plan_path=trial.B/'c8_batched_http_v2/prereg-v1/plan.json'
        self.assertEqual(m.sha(plan_path),'5290ddc18239d422e846f76ff4fa222b48188c6fb57bae68ea5119ec3bd9066c')
        plan=formal.read(plan_path);capture=formal.read(formal.pinned(plan['specs'][0]))
        base=formal.read(trial.B/'gdn_graph_combined_formal/prereg-formal-v1/specs/r1-c8-off.json')
        self.assertEqual(trial.identity(base,'on',server,server.source_pins(),qualification=True),capture['identity'])
        self.assertEqual(base['fixtures'],capture['fixtures']);self.assertEqual(base['environment'],capture['environment'])

    def test_greedy_difference_is_descriptive_not_invalid_timing_or_arithmetic_failure(self):
        points,ids=fixture();ids['r1-c8-on']['formal'][6]['raw_ids']=[7,99]
        ids['r1-c8-on']['formal'][6]['committed_ids']=[7,99]
        result=m.reduce_pairs(points,ids,PHASES,m.DECISION)
        self.assertTrue(result['speed_gate_passed']);self.assertFalse(result['all_pair_output_ids_identical'])
        self.assertEqual(result['output_comparison_status'],'DIFFERENT_GREEDY_TRAJECTORIES')
        self.assertFalse(result['numerical_error_inferred'])
        delta=result['pairs'][0]['output_comparison']['formal']
        self.assertEqual(delta['different_trajectories'],1)
        self.assertEqual(delta['first_difference']['raw_index'],1)
        self.assertEqual((delta['first_difference']['left_token'],delta['first_difference']['right_token']),(8,99))

    def test_missing_counts_order_and_baseline_drift_are_separate_gates(self):
        for change in ('missing','order','requests','ids'):
            points,ids=fixture()
            if change=='missing':points.pop()
            if change=='order':points[0],points[1]=points[1],points[0]
            if change=='requests':points[0]['requests']=63
            if change=='ids':ids[PHASES[0]]['formal'].pop()
            with self.assertRaises(ValueError):m.reduce_pairs(points,ids,PHASES,m.DECISION)
        points,ids=fixture();points[-1]['aggregate_output_tok_s']=105.
        result=m.reduce_pairs(points,ids,PHASES,m.DECISION)
        self.assertFalse(result['speed_gate_passed']);self.assertTrue(result['all_pair_output_ids_identical'])
        self.assertGreater(result['off_rate_spread_fraction'],.02)


if __name__=='__main__':unittest.main(verbosity=2)
