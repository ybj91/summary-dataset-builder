# Offline tests

Run `python -m unittest discover -s tests -v` after an editable installation.
For one module, use `python -m unittest tests.test_generation -v`; for one method,
use `python -m unittest tests.test_generation.GenerationTests.test_missing_usage_stops_and_reserves_unknown_billing -v`.

HTTP and streaming sources are mocked. Synthetic fixtures exercise preparation,
provenance, near-duplicate exclusion, budget reservations, retry exhaustion,
uncertain billing, interruption/reconciliation, review gates, and token-checked
exports. No test reads a credential file, downloads a dataset, or spends API credit.
Tests of the real local tokenizer skip unless the `tokenizer` extra is installed.

For the opt-in 10k durability regression in PowerShell:
`$env:RUN_SCALE_TESTS='1'; python -m unittest tests.test_scale -v`.
It uses synthetic records and mocked completions, exercises both scaling gates
and resume, and asserts exactly 10,000 training plus 200 validation/200 test
exported pairs. Remove the variable afterwards with `Remove-Item Env:\RUN_SCALE_TESTS`.
