"""Check archival joins and event denominators without rerunning flight models."""

import unittest

from scripts.part06.summarize_benchmarks import summarize_rows


def row(case, model, error, truth="goal_inside", predicted="goal_inside", replicate=0):
    return dict(case_id=str(case), scenario="reference", model=model, sigma_m="0.005",
                replicate=str(replicate), obs_eligible="True", truth_event=truth,
                prediction_event=predicted, error_m="" if error is None else str(error),
                time_error_s="" if error is None else "0.001")


class ArchiveSummaryTests(unittest.TestCase):
    def test_common_subset_matches_replicates_and_excludes_absent_crossings(self):
        records = [row(0, "A", 1), row(0, "B", 3),
                   row(0, "A", 9, replicate=1), row(0, "B", None, replicate=1),
                   row(1, "A", None, predicted="fit_failure"), row(1, "B", 7)]
        result = summarize_rows(records, ("A", "B"))["0.005"]
        self.assertEqual(result["common_crossings"], 1)
        self.assertEqual(result["models"]["A"]["common_position_m"]["median"], 1)
        self.assertEqual(result["models"]["A"]["paired_crossings"], 2)
        self.assertEqual(result["models"]["A"]["trials"], 3)
        self.assertEqual(result["models"]["A"]["event_accuracy"], 2 / 3)
        self.assertEqual(result["models"]["A"]["confusion"]["goal_inside"]["fit_failure"], 1)

    def test_duplicate_prediction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_rows([row(0, "A", 1), row(0, "A", 2)], ("A",))

    def test_missing_model_trial_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trial keys"):
            summarize_rows([row(0, "A", 1), row(1, "B", 2)], ("A", "B"))


if __name__ == "__main__":
    unittest.main()
