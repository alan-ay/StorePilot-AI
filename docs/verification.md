# Release verification — 0.2.0

Checked on 2026-09-05 with Python 3.12.13. Dependency versions used are listed in
`constraints-tested.txt`.

| Check | Result |
| --- | --- |
| Unit, integration and Streamlit interaction suite | 36 tests passed |
| Python warnings promoted to errors | Test process passed |
| Ruff lint and format | Passed |
| Python source compilation | Passed |
| Editable installation and dependency consistency | Passed using locally available dependencies |
| Wheel build | Passed; dashboard and shared stylesheet present |
| CLI execution | Forecasts, recommendations, draft orders and summary produced |
| Git staging whitespace check and source ZIP integrity | Passed |

The interaction suite executes the application using Streamlit AppTest. It
checks all seven pages, no retraining during navigation, single-store filtering,
empty search results, revised purchase quantities, approval gating and repeated
approval, scenario isolation, preference changes, sample-data switching, and
incomplete upload handling.

Business tests cover optional input fields, duplicate records, invalid values,
case packs and minimum quantities, inventory transfer capacity, approval snapshots,
persistence after reconnecting, cross-store budgets and isolation between review
batches. CSV export handling includes spreadsheet formula prefixes.

The Streamlit test harness logs a missing ScriptRunContext notice from its main
thread. This is a harness message, not an application exception. Its temporary
directory is explicitly cleaned up when the interaction test module completes.

## Boundaries of these checks

No real-browser screenshot, mobile visual, or screen-reader inspection was
performed. The CSS includes a narrow-screen layout and focus styling, but AppTest
does not verify rendered pixels. The Docker image was not built. No supplier or
POS integration, production deployment, real sales pilot, or forecast accuracy
claim is included in this verification.
