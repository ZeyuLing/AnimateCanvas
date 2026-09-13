<p align="center">
  <a href="https://zeyuling.github.io/AnimateCanvas/#demo"><img src="assets/readme/animatecanvas-cover.png" width="100%" alt="AnimateCanvas — Your cues. One coherent motion. Watch the complete demonstration." /></a>
</p>

<h3 align="center">Learning Implicit Motion Planning from Composable Kinematic Cues</h3>

<p align="center">
  <b><a href="https://zeyuling.github.io/AnimateCanvas/">Project page</a></b> &nbsp; / &nbsp;
  <b><a href="https://zeyuling.github.io/AnimateCanvas/#demo">Full video</a></b> &nbsp; / &nbsp;
  <b><a href="#benchmark-gallery">Benchmark gallery</a></b> &nbsp; / &nbsp;
  <b><a href="#resources">Paper &amp; code</a></b> &nbsp; / &nbsp;
  <b><a href="#hugging-face">Hugging Face</a></b>
</p>

---

**Compose kinematic cues. Generate coherent motion.** AnimateCanvas connects
cue-prescribed partial states into a coherent full-body trajectory. A shared flow-matching
model generates motion from position and rotation cues, with optional language and input
motion for editing. Cue imputation preserves the specified canvas values throughout generation.

## Watch AnimateCanvas

The complete **2:38 narrated film** combines kinematic control, motion editing, and character
animation. Explore individual scenes below, or watch it end to end.

<p><b><a href="https://zeyuling.github.io/AnimateCanvas/#demo">▶ Full film with chapters</a></b> &nbsp; · &nbsp; <a href="https://zeyuling.github.io/AnimateCanvas/assets/media/animatecanvas-demo-1080p.mp4">1080p MP4</a> &nbsp; · &nbsp; <a href="https://github.com/ZeyuLing/AnimateCanvas/releases/download/animatecanvas-demo-20260913/animatecanvas-demo-1440p.mp4">1440p original</a></p>

<table>
  <tr>
    <td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/route-v56.mp4"><img src="assets/showcase/route-v56.webp" alt="Routes &amp; local targets" width="100%" /></a><br /><b>Routes &amp; local targets</b></td>
    <td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/footsteps-animatecanvas.mp4"><img src="assets/showcase/footsteps-v56.webp" alt="Footsteps &amp; heading" width="100%" /></a><br /><b>Footsteps &amp; heading</b></td>
    <td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/jump-v56.mp4"><img src="assets/showcase/jump-v56.webp" alt="Key poses &amp; trajectories" width="100%" /></a><br /><b>Key poses &amp; trajectories</b></td>
  </tr>
  <tr>
    <td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/editing-v56.mp4"><img src="assets/showcase/editing-v56.webp" alt="Language-guided editing" width="100%" /></a><br /><b>Language-guided editing</b></td>
    <td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/boxing-v56.mp4"><img src="assets/showcase/boxing-v56.webp" alt="Composed hand controls" width="100%" /></a><br /><b>Composed hand controls</b></td>
    <td width="33%" align="center"><a href="https://zeyuling.github.io/AnimateCanvas/assets/showcase/basketball-animatecanvas.mp4"><img src="assets/showcase/basketball-v56.webp" alt="Timed spatial targets" width="100%" /></a><br /><b>Timed spatial targets</b></td>
  </tr>
</table>

## Benchmark gallery

**24 AnimateCanvas inference cases — four per benchmark.** The six collections cover
temporal control, body-part control, sequential generation, instruction editing,
style–content editing, and text-to-motion.

Each preview links to its full clip and source viewer. For edits, **input motion is on the left**
and **AnimateCanvas is on the right**. Temporal previews distinguish supplied poses (amber) from generated motion (blue).
The orange marker identifies the controlled wrist. [Color guide and complete previews →](https://zeyuling.github.io/AnimateCanvas/#benchmarks)

<table>
<tr>
<td width="50%" valign="top"><a href="https://zeyuling.github.io/AnimateCanvas/#case-prediction"><img src="assets/benchmarks/prediction.webp" width="100%" alt="Temporal control preview" /></a><br /><b>Temporal control</b><br /><a href="https://zeyuling.github.io/AnimateCanvas/?benchmark=temporal#benchmarks">Watch all 4 cases →</a></td>
<td width="50%" valign="top"><a href="https://zeyuling.github.io/AnimateCanvas/#case-body-part-reach"><img src="assets/benchmarks/body-part-reach.webp" width="100%" alt="Body-part control preview" /></a><br /><b>Body-part control</b><br /><a href="https://zeyuling.github.io/AnimateCanvas/?benchmark=body#benchmarks">Watch all 4 cases →</a></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://zeyuling.github.io/AnimateCanvas/#case-sequential"><img src="assets/benchmarks/sequential.webp" width="100%" alt="Sequential generation preview" /></a><br /><b>Sequential generation</b><br /><a href="https://zeyuling.github.io/AnimateCanvas/?benchmark=sequential#benchmarks">Watch all 4 cases →</a></td>
<td width="50%" valign="top"><a href="https://zeyuling.github.io/AnimateCanvas/#case-instruction"><img src="assets/benchmarks/instruction.webp" width="100%" alt="Instruction editing preview" /></a><br /><b>Instruction editing</b><br /><a href="https://zeyuling.github.io/AnimateCanvas/?benchmark=instruction#benchmarks">Watch all 4 cases →</a></td>
</tr>
<tr>
<td width="50%" valign="top"><a href="https://zeyuling.github.io/AnimateCanvas/#case-style"><img src="assets/benchmarks/style.webp" width="100%" alt="Style–content editing preview" /></a><br /><b>Style–content editing</b><br /><a href="https://zeyuling.github.io/AnimateCanvas/?benchmark=editing#benchmarks">Watch all 4 cases →</a></td>
<td width="50%" valign="top"><a href="https://zeyuling.github.io/AnimateCanvas/#case-text-to-motion"><img src="assets/benchmarks/text-to-motion.webp" width="100%" alt="Text-to-motion preview" /></a><br /><b>Text-to-motion</b><br /><a href="https://zeyuling.github.io/AnimateCanvas/?benchmark=text#benchmarks">Watch all 4 cases →</a></td>
</tr>
</table>

<p align="center"><b><a href="https://zeyuling.github.io/AnimateCanvas/#benchmarks">Browse all cases with task filters →</a></b></p>

## One canvas, coherent completion

<img src="assets/figures/animatecanvas-pipeline.png" alt="AnimateCanvas pipeline: compose kinematic cues on the canvas, predict motion with a shared MMDiT, preserve cue values through imputation, and train coherent completion with geometric and transition objectives." width="100%" />

- **Compose the cues.** Key poses, trajectories, local position and rotation targets share a
  time–kinematic-variable canvas.
- **Plan the motion.** One shared flow model connects cue-prescribed partial states into a
  coherent full-body trajectory; cue imputation preserves specified canvas values.
- **Edit in context.** Language and input motion support changes to an existing action.

## Resources

| Resource | Status |
| :--- | :--- |
| Project page & video | [Explore AnimateCanvas](https://zeyuling.github.io/AnimateCanvas/) |
| Paper & citation | Coming soon |
| Training & inference code | Coming soon |
| Hugging Face model | Coming soon |
| Benchmark viewers | Available below |

### Hugging Face

[Temporal control](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) ·
[Body-part control](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) ·
[Sequential generation](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) ·
[Instruction editing](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard) ·
[Style & content editing](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) ·
[Text-to-motion](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard)

Follow this repository for the paper, model, and dedicated code release.

---

<sub>[Media provenance](assets/media-manifest.json) · [Media notes](MEDIA.md) · [Motius](https://github.com/ZeyuLing/Motius)</sub>
