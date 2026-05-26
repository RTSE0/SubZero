<h1 align="center">🥶 SUBZERO</h1>

<h3 align="center"> A normal way to get normals, entirely on your machine.</h3>

Built as a free alternative to [Switchlight Studio](https://www.beeble.ai/) after it turned into a subscription based model. Runs entirely on your machine, no need to pay $504 a year. This pipeline is designed to extract temporally stable surface normal maps for raw video and image sequences.

Given a keyed portrait video (PNG/EXR sequence with alpha), SubZero extracts **normal maps** — surface geometry data used for relighting in Blender, Nuke, or After Effects.
By chaining Marigold with BiSeNet and 3D spatial geometry trackers (MediaPipe), SubZero overcomes the traditional edge-bleeding, flickering, and specular hallucination limitations inherent to modern diffusion models.

---
## Demo
> Normal map used to relight footage, full quality mode (ensemble = 12, res = 1024)
<p align="center">
  <img width="800" height="450" alt="Testfootage-ezgif com-optimize" src="https://github.com/user-attachments/assets/d4c6083d-71f3-4ed0-bb85-99e70d6b058b" />
</p>

---

## How It Works
 
SubZero pipelines several open-source models together:
 
| Stage | Model | What it does |
|-------|-------|-------------|
| Normal extraction | [Marigold](https://marigoldmonodepth.github.io/) | Diffusion-based surface normal estimation |
| Face segmentation | [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) | Per-region facial parsing (19 regions) |
| Head pose | [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | 3D head rotation matrix via face landmark tracking |

---

## Getting Started

**Requirements:** Python 3.10+, CUDA GPU recommended (tested on RTX 5070)
```bash
# Clone repo
git clone https://github.com/RTSE0/SubZero
cd SubZero
 
# Install dependencies
pip install -r requirements.txt
 
# Clone BiSeNet face parser (required for per-region filtering)
git clone https://github.com/zllrunning/face-parsing.PyTorch
```
### requirements.txt
```
numpy
torch
opencv-contrib-python
pillow
rembg[gpu]
mediapipe
diffusers
transformers
accelerate
```
> BiSeNet and MediaPipe model weights download automatically on first run (~50MB and ~6MB respectively). You can also remove the [gpu] in rembg

## Input Format
 
SubZero is designed for **pre-keyed footage** — export your subject with alpha from DaVinci Resolve, Nuke, or After Effects before running.
 
| Input | Quality | Notes |
|-------|---------|-------|
| PNG sequence with alpha | ✓ Best | Use Magic Mask, Delta Key, or rotoscope in Resolve |
| EXR sequence with alpha | ✓ Best | Preferred for linear-gamma VFX pipelines |
| MP4 / MOV | ⚠ Fallback | rembg used for background removal — lower quality around hair |
 
**Recommended Resolve export:** File → Export → Individual Clips → PNG, enable Export Alpha.
 
---
## Usage
 
```bash
# Full quality
python subzero_normals.py "./frames_folder" "./output_normals"
 
# Fast preview
python subzero_normals.py "./frames_folder" "./output_normals" --preview
```
 
### Quality flags
| Flag | Default | Preview |
|------|---------|---------|
| `--steps` | 10 | 4 |
| `--ensemble` | 12 | 4 |
| `--res` | 1024 | 512 |
| `--no-temporal` | — | disables temporal smoothing |
 
---
## Output
 
```
output_normals/
  normal_0001.png   # RGBA — normal map with alpha carried through
  normal_0002.png
  ...
```
 
All output frames carry the original alpha channel. Import as image sequence in Resolve, Nuke, or After Effects at your source FPS.
 
---
## Limitations
 
- Optimised for forward-facing subjects. Quality degrades on extreme profile angles — this is a limitation of monocular normal estimation in general, not specific to SubZero.
- SwitchLight uses proprietary models trained on lightstage capture data. SubZero is a best-effort approximation using open models.
---
 
## Acknowledgements
 
- [Marigold](https://marigoldmonodepth.github.io/) — Ke et al., CVPR 2024
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) — BiSeNet face parser
- [MediaPipe](https://ai.google.dev/edge/mediapipe) — Google
---
 
## License
 
MIT
