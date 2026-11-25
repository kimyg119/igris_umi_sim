# Report figure sources

Run from the repository root. Images, arrays, and provenance reports are written
under ignored `outputs/report_figures/`; the source inputs are left unchanged.

```bash
python -m igris.hl_ik.report_figures.make_figures
python -m igris.hl_ik.report_figures.make_pose_comparison
python -m igris.hl_ik.report_figures.make_bent_comparison
python -m igris.hl_ik.report_figures.render_pelvis
```

- `make_figures`: hierarchy and skill-flow diagrams; requires Matplotlib and the
  system Noto Sans CJK font.
- `make_pose_comparison` / `make_bent_comparison`: render saved EE-only versus
  human-pose-prior IK results. Requires the HL-IK environment (NumPy, SciPy,
  MuJoCo, Pillow, and solver dependencies), the generated model under
  `outputs/hl_ik_20260915/model/`, and the saved
  `outputs/hl_ik_pipeline_20260915/final/` experiment. The checkpoint and collision
  envelope referenced by that experiment must also be available. Source hashes,
  matching targets, and result validity are checked before rendering.
- `render_pelvis`: highlights the actual `pelvis` link using the generated
  `reference_pelvis.xml`, model provenance, and repository URDF. Requires NumPy,
  SciPy, MuJoCo, and Pillow.

The saved experiments, trained checkpoint, and generated models are local inputs,
not included in this source snapshot. See the parent [README](../README.md) for
model construction and experiment commands. Robot images use forward kinematics
without stepping physics or sending robot commands; they are visualizations,
not execution validation.
