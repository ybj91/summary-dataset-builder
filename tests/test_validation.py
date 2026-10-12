import json
import unittest

from summary_dataset_builder.validation import screening_flags, validate_teacher


class ValidationTests(unittest.TestCase):
    def test_acceptance_and_rejection_contract(self):
        for value in (
            {"accepted": True, "summary": "Supported summary.", "rejection_reason": ""},
            {"accepted": False, "summary": "", "rejection_reason": "Incomplete source"},
        ):
            self.assertEqual(validate_teacher(json.dumps(value), "stop").accepted, value["accepted"])

    def test_rejects_invalid_or_inconsistent_responses(self):
        invalid = [
            "not-json", "[]", "null",
            '{"accepted":true,"summary":"x","rejection_reason":"","extra":1}',
            '{"accepted":1,"summary":"x","rejection_reason":""}',
            '{"accepted":true,"summary":"","rejection_reason":""}',
            '{"accepted":false,"summary":"x","rejection_reason":"bad"}',
            '{"accepted":false,"summary":"","rejection_reason":" "}',
            '{"accepted":true,"summary":"x","rejection_reason":" "}',
            '{"accepted":true,"accepted":false,"summary":"","rejection_reason":"bad"}',
        ]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_teacher(content, "stop")

    def test_length_limited_json_is_not_a_success(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_teacher('{"accepted":false,"summary":"","rejection_reason":"bad"}', "length")

    def test_screening_numbers_and_instruction_contamination(self):
        self.assertEqual(screening_flags("It cost $420,000.", "It cost $420000."), [])
        self.assertIn("unsupported_number", screening_flags("It cost $420,000.", "It cost $4 million."))
        self.assertIn(
            "instruction_or_meta_content",
            screening_flags("An injected note says ignore prior directions.", "The note says ignore prior directions."),
        )

    def test_no_minimum_word_count_that_forces_padding(self):
        self.assertEqual(screening_flags("The source is short.", "A short source."), [])
