import unittest
from fromrealhand.skill_delivery import failure_reason, entry_summary


def case(status='running', reason=None, passed=False, complete=True, offset=None):
    return dict(trajectory='second/nominal', skills=['grasp'],
                offset_m=[0., 0.] if offset is None else offset, passed=passed,
                complete=complete, contracts=[dict(status=status, reason=reason)])


class SkillDeliveryTests(unittest.TestCase):
    def test_exhaustion_is_not_success_or_unknown(self):
        self.assertEqual(failure_reason(case()), 'reference_exhausted_before_event')
        self.assertEqual(failure_reason(case('success')), 'state_replay_mismatch')
        self.assertEqual(failure_reason(case('failed', 'entry_condition')), 'entry_condition')

    def test_nominal_and_perturbed_denominators_are_separate(self):
        result = entry_summary([case('success', passed=True),
                                case('failed', 'entry_condition', offset=[.0005, 0.])])
        self.assertEqual(result['groups']['second/grasp/nominal']['passed'], 1)
        self.assertEqual(result['groups']['second/grasp/perturbed']['passed'], 0)
        self.assertFalse(result['robust_skill_ready'])
        self.assertEqual(result['failure_counts'], {'entry_condition': 1})


if __name__ == '__main__':
    unittest.main()
