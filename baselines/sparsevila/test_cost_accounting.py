"""Regression checks for SparseVILA's pre- and post-compaction cost."""

import unittest

from baselines.evaluation.run_mask import _cost_record


class SparseVILACostAccountingTest(unittest.TestCase):
    def cost(self, answer_tokens, first_generation_tokens=None):
        return _cost_record(
            n_vis=1024,
            kept_tokens=102,
            prefill_tokens=133,
            first_generation_tokens=first_generation_tokens,
            total_layers=36,
            n_gen_tokens=answer_tokens,
            shared_setup_sec=0.0,
            shared_question_sec=0.0,
        )

    def test_one_visible_token_is_charged_before_compaction(self):
        cost = self.cost(1, first_generation_tokens=133)
        self.assertEqual(cost["prefill_tl"], 4788.0)
        self.assertEqual(cost["gen_tl"], 4788.0)
        self.assertEqual(cost["total_tl"], 9576.0)
        self.assertEqual(cost["n_vis_first_generation"], 133.0)
        self.assertEqual(cost["n_vis_decode"], 102.0)

    def test_later_tokens_use_compacted_cache(self):
        cost = self.cost(3, first_generation_tokens=133)
        self.assertEqual(cost["gen_tl"], 36.0 * (133 + 2 * 102))

    def test_other_methods_keep_uniform_generation_cost(self):
        cost = self.cost(3)
        self.assertEqual(cost["gen_tl"], 36.0 * 3 * 102)

    def test_empty_decoded_answer_has_no_generation_charge(self):
        cost = self.cost(0, first_generation_tokens=133)
        self.assertEqual(cost["gen_tl"], 0.0)


if __name__ == "__main__":
    unittest.main()
