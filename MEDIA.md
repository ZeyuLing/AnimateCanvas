# Media notes

## Official credited edition — September 18, 2026

The [formal release](https://github.com/ZeyuLing/AnimateCanvas/releases/tag/animatecanvas-official-demo-20260918)
adds the final author list and affiliations to the opening and appends a nine-second
project/Motius end card. The original motion demonstrations and chapter timing are
retained. It provides a 2560×1440 English master and a 1920×1080 Chinese-captioned
edition, both 30 fps and 2:47 long. These large videos are release assets, not code
repository blobs. The embedded lightweight 2:38 website player remains the earlier
edition; its adjacent link leads to the new credited editions.

## Existing website demonstrations

The current release is **AnimateCanvas**. The September 13 branding update
renames the opening, burned-in captions, three English voice takes, README cover,
and editing-preview labels. The accepted V56 motion, camera, scene timing,
anatomical keyframe cues, post-dunk support corrections, and music are retained.
Other voice takes are reused; benchmark case choices, motion data, frame counts,
and playback rates are unchanged. Legacy viewer method keys are retained for
working links to the original experiment viewers.

The narrated film is the September 2026 editorial update: the current paper title,
Figure 2 and terminology replace the earlier introduction and narration. The V56 complete film also incorporates anatomical pose-cue skeletons, corrected
vault-rail heights, and post-dunk planted-foot support. The current branding edition retains the authored
camera, timing, trajectory-completion effects and music, with three renamed voice takes. The six
character clips are excerpts from this updated full demonstration.
The website does not present these edited presentation clips as raw benchmark
measurements.

The benchmark gallery contains 24 persisted AnimateCanvas inference cases from
Motius, with four examples in each of six benchmark collections: temporal control,
body-part control, BABEL sequential generation, instruction editing, style–content
editing, and text-to-motion. The style–content collection includes two style and
two content edits. The temporal collection covers prediction, in-betweening,
nonuniform keyframes, and continuation without text. The gallery does not include
corruption or repair previews.

The featured local-control and sequential previews are HumanML3D case `001014`
and BABEL case `val_6604`. Their complete frame sequences were inspected for
motion continuity and input correspondence before selection; these are curated
examples, not a substitute for the aggregate benchmark results.

The September 2026 expansion screened 37 candidate cases using viewer previews.
Final non-editing candidates underwent per-frame joint-continuity checks and
enlarged inspection around detected changes; editing candidates were inspected
side by side across the complete action. A wrist-pulling local-control case and
an ambiguous opposite-leg edit were excluded. These checks guide visual selection,
not a new benchmark score. Original benchmark motions were retained without
smoothing or joint corrections.

The previews use each benchmark viewer's SMPL decoder and original motion frames.
No motion post-processing is applied. Camera, materials, and framing are standardized;
editing comparisons use input and output side by side. Mesh frustum culling is
disabled when capturing the editing viewers so moving figures remain visible.
Original playback rates are retained (20 fps for the local-control cases and
30 fps for the other selected cases). GIFs are short 12 fps previews; the MP4s
contain the complete selected cases at their source playback rate.

Case identifiers, input conditions, source viewers, and MP4 hashes are recorded in
[the media manifest](assets/media-manifest.json). For the editing viewers, select
the case index and track recorded there; these viewers do not expose case-specific
deep links. The character film and benchmark outputs are distinct media collections.

The README cover retains the established editorial design. The six character
stills use the V56 picture-only frames without narration subtitles. Earlier
versioned masters and release assets remain available in the release history.

The method figure is Figure 2 from paper revision `95c4fb06`. It is exported
directly from the paper PDF, rather than reconstructed for the website.

Character designs, body models, and other third-party assets retain their
respective rights and licenses. Rendered previews do not grant permission to
redistribute the underlying character meshes, textures, or body-model files;
those source assets are not included in this repository.

The 1080p web film and short MP4s are served by the project page. The full 1440p
master is distributed separately as a GitHub Release asset to keep large
production masters out of Git history. Neither model weights nor training
data are included in this website repository.
