# Attribution

`chewing-vision` is licensed under the MIT License (see `LICENSE`).

It integrates and depends on the following third-party software.

## orofacIAnalysis

- **Version**: 0.1.2
- **Author**: Cameron Maloney (cameron.maloney@warriorlife.net)
- **License**: MIT
- **PyPI**: https://pypi.org/project/orofacIAnalysis/0.1.2/
- **Usage**: Imported as a runtime dependency. The `OrofacEngine` in
  `chewing/engines/orofac.py` wraps `orofacIAnalysis.ChewAnnotator` and
  reuses its `Cycle` / smoothing utilities for chewing analysis. No source
  code from orofacIAnalysis is vendored into this repository.

## MediaPipe

- **License**: Apache License 2.0
- **Source**: https://github.com/google-ai-edge/mediapipe
- **Usage**: `mediapipe.tasks.python.vision.FaceLandmarker` is used to
  extract face landmarks and blendshapes (including `jawOpen`).

## Other dependencies

`opencv-python`, `numpy`, `scipy`, `matplotlib`, `pandas`, `Pillow` are used
under their respective licenses (BSD-style / MIT).
