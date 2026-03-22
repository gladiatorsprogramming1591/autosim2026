
REBUILT simulator semi-rewrite package

Included files
- shared_sim_core.py
- shared_field_visuals.py
- pathplanner_compat.py
- path_editor.py
- debug_tuner.py
- batch_path_heatmaps.py

Main changes
- Native path loader now accepts both native .json files and PathPlanner .path files.
- Added PathPlanner subset export helper.
- Added per-run contact vs capture traces.
- Added richer debug / batch reporting.
- Path editor can now export directly to PathPlanner with the P key.
- Path loader cycles through both .json and .path files.

Notes
- PathPlanner compatibility is intentionally subset-based.
- Advanced PathPlanner data is preserved in metadata where practical, but not all features affect the simulator.
- Your native JSON format still works and remains the simplest source-of-truth format.

Suggested install
1. Back up your current scripts.
2. Replace the old .py files with these versions.
3. Keep your existing params.json and image assets.
4. Test path_editor.py first, then debug_tuner.py, then batch_path_heatmaps.py.
