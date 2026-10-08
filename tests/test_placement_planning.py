import unittest
from fromrealhand.desktop.placement_planning import placement_plan, INSTRUCTIONS


class PlacementPlanningTests(unittest.TestCase):
    def test_explicit_instructions_include_prerequisites_and_return(self):
        for text in INSTRUCTIONS:
            for scene in ('first','second'):
                plan=placement_plan(text,scene)
                self.assertEqual(plan['skills'],['reach','grasp','lift','transport','place','return_home'])

    def test_existing_transport_and_unsafe_commands_not_reinterpreted(self):
        for text in ('把杯子搬到目标位置','直接松手','把杯子扔下去','把杯子放到我手上',
                     '不要把杯子放到目标位置并返回起点','把杯子放到目标位置并返回起点，然后倒水'):
            self.assertIsNone(placement_plan(text,'first'))
