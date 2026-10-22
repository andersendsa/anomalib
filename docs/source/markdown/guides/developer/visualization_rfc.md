# RFC: Standalone visualization package

Status: **Proposed; implementation requires maintainer approval.**

Related issue: [#3872](https://github.com/open-edge-platform/anomalib/issues/3872).
This proposal inventories the existing behavior and defines the decisions to review
before implementing the package. API examples below are proposals, not available APIs.

## Problem and scope

Library visualization imports the training stack and returns Pillow images, while the
application renders NumPy frames using OpenCV and transports colored PNG overlays.
Both implement color mapping, alpha blending, resizing, and text rendering. Moving
these operations into a shared renderer must preserve inference values and existing
consumer behavior, rather than force every consumer to use the same presentation.

The first release covers image rendering, including individual video frames. Training,
inference, post-processing, file access, image encoding, streaming, and browser display
remain consumer responsibilities. No plugin discovery service, serialized layout
language, GPU renderer, or new frontend controls are proposed.

## Existing behavior and integration inventory

Paths below are relative to the repository root.

| Surface                                                                                                                             | Existing behavior                                                                                                                                                       | Proposed responsibility                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `src/anomalib/visualization/image/functional.py`                                                                                    | Image conversion, color maps, optional display normalization, alpha overlays, binary/filled/contour masks, text/font sizing, image grids; imports Torch and torchvision | Extract array/Pillow rendering; keep tensor and path conversion in library adapters              |
| `src/anomalib/visualization/image/item_visualizer.py`                                                                               | Field selection, ordered overlays, titles, configurable field sizes, ground truth and prediction fields                                                                 | Translate existing field configuration into shared panels                                        |
| `src/anomalib/visualization/image/visualizer.py`                                                                                    | Single-item and batch handling; Pillow return values; test/predict callbacks and output filenames                                                                       | Preserve public signatures, return shapes, hooks, and saving in the library                      |
| `src/anomalib/visualization/base.py`                                                                                                | Lightning `Callback` base class                                                                                                                                         | Remain in the library; never imported by the standalone package                                  |
| `src/anomalib/models/components/base/anomalib_module.py`                                                                            | Constructs and configures the default library visualizer                                                                                                                | Continue using the compatibility adapter                                                         |
| `src/anomalib/pipelines/tiled_ensemble/components/visualization.py`                                                                 | Renders merged predictions and saves per-item results                                                                                                                   | Reuse the library adapter after tile merging                                                     |
| `application/backend/src/services/model_service.py`                                                                                 | Converts the anomaly map to JET RGBA; alpha is proportional to anomaly intensity; encodes base64 PNG; returns label and score                                           | Shared color mapping; retain inference and transport encoding                                    |
| `application/backend/src/utils/visualization.py`                                                                                    | Decodes RGBA overlay, resizes, blends at alpha 0.5, draws label and integer percentage; logs failures and continues                                                     | Adapt decoded arrays to shared components; retain legacy failure policy                          |
| `application/backend/src/workers/inference.py`                                                                                      | Selects heatmap visibility, always draws label, produces visualized stream frames                                                                                       | Select components; retain worker and stream ownership                                            |
| `application/ui/src/features/inspect/dataset/media-preview/inference-result/inference-result.component.tsx`                         | Displays encoded heatmap with user-controlled opacity                                                                                                                   | Keep PNG response schema and browser interaction                                                 |
| `src/anomalib/callbacks/visualizer.py`, `src/anomalib/deploy/inferencers/base_inferencer.py`, `tests/unit/utils/test_visualizer.py` | References to the older `anomalib.utils.visualization` namespace                                                                                                        | Audit historical compatibility separately; do not claim these references establish a working API |

Library defaults currently select image and ground-truth mask fields plus image/map
and image/predicted-mask overlays, at 256 by 256 pixels per field. Ground-truth
contours are white; predicted contours are red. Anomaly-map normalization defaults
to false. The application instead uses one frame with a score-weighted heatmap and
label. Preserve these as separate consumer presets built from shared components.

Bounding boxes are a required new component, not an existing feature of the inspected
image functional module. Ground truth, plain masks, titles, and configurable fonts
must remain available even though the illustrative issue API does not show them.

Integration also reaches package exports, the root and backend `pyproject.toml`
files and lockfiles, application build/install paths, visualization reference docs,
the visualization notebook, and release notes. Relevant existing checks include
`tests/unit/visualization/image/test_functional.py`,
`tests/integration/utils/test_visualizer.py`,
`tests/unit/utils/callbacks/visualizer_callback/test_visualizer.py`, and
`application/backend/tests/unit/utils/test_visualization.py`. Existing legacy test
imports must be classified before treating those suites as migration evidence.

## Package boundary and distribution

Proposed distribution: `anomalib-visualization`; import: `anomalib_visualization`.
Place it under `packages/anomalib-visualization/` with its own `pyproject.toml`,
source directory, tests, README, license, and wheel/sdist build. Confirm name
availability and release ownership before publishing.

Use NumPy and Pillow as core dependencies. Retain the existing Pillow rendering
operations where possible; contours can reuse the existing mask logic. An optional
OpenCV extra supplies exact application JET, text, and contour behavior where needed
for compatibility. Do not require both GUI and headless OpenCV distributions in one
environment; reconcile this extra with the backend's existing dependency choice.
The extra must also remain independent of the training stack.

No dependency on Anomalib, Torch, torchvision, Lightning, OpenVINO, FastAPI, or
application schemas is permitted, including transitive dependencies. Dependency
direction is library/application to renderer only. Tensor conversion stays in the
library; base64 and response-model conversion stay in the application.

Keep the package independently buildable without converting the repository into a
uv workspace. Workspaces share a lockfile, which is unnecessary for this extraction
([uv workspace documentation](https://docs.astral.sh/uv/concepts/projects/workspaces/)).
Use local path sources for development and ordinary versioned requirements in
published metadata. Build and install the standalone wheel before updating consumer
lockfiles; never ship a wheel that requires a repository-relative path.

## Data contract and coordinate alignment

`Visualizer.render(image, prediction)` returns a newly allocated RGB `uint8` NumPy
array. Inputs are borrowed read-only and must never be modified. The core accepts
one HWC RGB image per call, not batches, tensors, filenames, or encoded images.
Adapters explicitly convert BGR, CHW, grayscale, RGBA, floating-point images, and
tensors; no channel-order or numeric-range guessing occurs in the core.

`Prediction` is a small typed record with optional fields:

| Field                  | Contract                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------- |
| `anomaly_map`          | Finite floating-point array `(H, W)`; original inference values                                         |
| `pred_mask`, `gt_mask` | Boolean arrays `(H, W)`; decisions supplied by the caller                                               |
| `boxes`                | Finite array `(N, 4)` of image-space `(x_min, y_min, x_max, y_max)` coordinates; exclusive upper bounds |
| `label`                | Display string supplied by the caller, independent of score                                             |
| `score`                | Finite scalar, displayed without recalibration or changing the label                                    |
| `heatmap_rgba`         | Optional precolored `uint8` `(H, W, 4)` overlay for the existing application transport                  |

All spatial fields must already be aligned with the input image. Mismatched sizes
raise an error rather than triggering an inferred resize. Adapters must explicitly
undo preprocessing: remove letterbox padding, resize maps with bilinear interpolation,
resize masks with nearest-neighbor interpolation, and transform boxes using the same
scale and crop offsets. A crop-local prediction must be rendered against that crop
or explicitly placed back in original-image coordinates. Missing crop metadata is
an error; equal array dimensions alone do not establish alignment.

The first release supports aligned inputs instead of a general transform graph.
Layout resizing happens after annotations and therefore resizes image and annotations
together. Crop/resize fixtures with known landmarks must verify adapter alignment.

Heatmaps have an explicit display range, default `(0, 1)`. Values outside that range
are clipped on a temporary display array only. An optional min/max display mode
preserves the library's existing customization, including a defined constant-map
result. No annotation thresholds scores, derives masks, or changes labels. NaN and
infinity are rejected. Scores outside `[0, 1]` are valid raw scores; percentage
formatting is an explicit consumer choice and never clamps the original score.

Raw-map and precolored-overlay rendering are separate components so a decoded PNG
cannot accidentally be normalized or colored twice. Invalid image shapes/dtypes,
unordered or out-of-bounds boxes, invalid opacity, and invalid display ranges raise
descriptive errors. Empty box sets and empty masks are valid.

## Composition and extension contracts

Proposed built-ins are `Heatmap`, `Mask`, `Contours`, `BoundingBoxes`, `Label`,
`Score`, and `RGBAOverlay`. Masks/contours can select `pred_mask` or `gt_mask`.
Style parameters cover opacity, colormap, RGB color, line thickness, font, position,
and score formatting. Drawing uses ordered source-over composition.

An annotator implements `render(image, prediction) -> RGBArray`. It receives the
current canvas and returns a same-sized canvas without modifying inputs. Built-ins
and user implementations use this same structural protocol; subclassing or registry
modification is unnecessary. A custom layout implements the same render signature
but may return a different image size. Validate returned arrays at composition
boundaries so invalid custom output fails at the responsible component.

`Overlay(*annotations)` applies annotations left to right. `OriginalImage()` returns
a copy. `Row(*panels)` and `Column(*panels)` compose nested panels with explicit
padding and background color; unequal panel sizes are padded, never stretched.
A grid is expressible as rows inside a column. Compatibility adapters can resize
panels explicitly to preserve the current library field-size behavior.

```python
# Proposed API; available only after implementation and release.
from anomalib_visualization import (
    Contours,
    Heatmap,
    Label,
    Mask,
    OriginalImage,
    Overlay,
    Row,
    Visualizer,
)

visualizer = Visualizer(
    layout=Row(
        OriginalImage(),
        Overlay(Heatmap(opacity=0.4, colormap="jet"), Label(show_score=True)),
        Overlay(Mask(), Contours(color="red", thickness=2)),
    ),
)
rendered = visualizer.render(image, prediction)
```

`Visualizer()` defaults to heatmap followed by label, skipping absent optional
fields. Explicit annotations require their fields and raise if absent. Passing
`annotations=[]` returns an image copy; `None` selects defaults. Supplying both
`annotations` and `layout` is an error. An explicitly empty row/column is invalid.
Custom components are ordinary Python objects passed directly into these layouts;
saving, displaying, and encoding the result happen outside the renderer.

## Compatibility and rollout

1. Approve this RFC and record the review link and accepted decisions here. Do not
   implement or publish the package before this step.
2. Build the independent package and its contract tests. Record baseline images
   from the existing library and application before replacing any rendering code.
3. Add a library adapter preserving exports, constructor options, custom field
   lookup behavior, Pillow returns, single/batch behavior, missing-field behavior,
   callbacks, and saved output naming. Tensor-to-CPU conversion occurs here once
   per item. Preserve legacy display normalization only when requested.
4. Add application adapters preserving the PNG/base64 response, score-weighted
   alpha, heatmap toggle, label styling, integer percentage, and RGB stream output.
   Keep logging and best-effort overlay failure recovery at this boundary, while
   the core remains strict. Both PNG generation and stream compositing use shared
   primitives; avoid introducing an extra encode/decode round trip.
5. Remove duplicated primitive implementations only after parity checks pass.
   Update docs, examples, dependency locks, installation/build paths, and changelog.

Existing public library APIs remain supported throughout the first release. No
removal date or warning is introduced by this RFC. A later deprecation requires
its own documented release policy. Existing saved predictions and application
response schemas require no migration. Users wanting the standalone API explicitly
convert their inputs and own output I/O; users retaining old APIs need no code change.

Compare dimensions, ordering, colors, opacity, masks, labels, and fonts against
baseline images. Require exact equality for integer primitives with a pinned backend;
use documented tolerances for antialiasing and font rasterization. A visually similar
replacement is not automatically compatible. Retain the necessary backend until
maintainers accept any deliberate rendering difference.

## Verification and acceptance criteria

- Core tests cover opacity endpoints, composition order, missing/invalid fields,
  mask modes, contours at image boundaries, empty inputs, boxes, raw scores, constant
  maps, and custom annotation/layout contracts. Assert all source arrays and scores
  remain unchanged, including after failed rendering.
- Deterministic synthetic fixtures verify RGB/BGR handling, score-weighted RGBA,
  crop placement, non-square resizing, nearest-neighbor masks, and box alignment.
- Install a built wheel into a fresh uv-managed environment outside the checkout.
  Render defaults, each annotation, and a composed layout. Inspect installed
  dependencies and imported modules to prove training/runtime packages are absent.
  Repeat for the optional OpenCV extra and build the wheel from the sdist.
- Run library functional tests, single-item/multi-item and callback workflows,
  tiled-ensemble output tests, backend visualization/service/worker tests, and an
  application inspect/stream smoke test. Verify the heatmap toggle leaves labels
  visible and preserves inference response values.
- Measure representative 256, 1080p, and multi-panel rendering against the existing
  paths, including allocations and frame latency. Agree on acceptable regression
  limits during review; do not invent performance claims before measurement.
- Add standalone installation, defaults, styling, custom component, layout,
  coordinate conversion, and migration examples to user and reference docs.

## Decisions requested from maintainers

- Approve the distribution name, location, release ownership, and supported Python
  versions shared with consumers.
- Approve NumPy/Pillow core plus optional OpenCV compatibility support, or require
  OpenCV in the core if maintaining two rasterizers is not justified by parity tests.
- Approve aligned RGB array inputs, strict core errors, and consumer-owned geometry
  restoration rather than implicit resizing.
- Approve separate consumer presets and the existing API retention policy.
- Agree on rendering tolerance and performance budgets before adapters replace code.

Approval record: pending. This RFC does not resolve or close the implementation epic.
