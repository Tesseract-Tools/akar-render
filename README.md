# akar-render

The photoreal renderer for [Akar](https://akar.tesseractstudio.co), the 3D modeller by Tesseract Studio.
It is a small program built on [Cycles](https://www.cycles-renderer.org) (the Blender Foundation's
path tracer, Apache 2.0). Akar writes the model to a scene file and runs `akar-render` in its own
process; the picture comes back as it improves.

This repository holds only the renderer's helper program (`host/`) and the build that compiles
it with Cycles for Windows (NVIDIA OptiX and CUDA, and the processor). Akar's own code is not here.

- `host/akar_render.cpp`: reads the scene file, builds the Cycles scene, renders, reports progress.
- `host/cycles-app-append.cmake`: adds the program to Cycles' build.
- `.github/workflows/build-windows.yml`: builds the render pack (Actions → Build render pack → Run).
- `docs/scene-format.md`: the scene file Akar writes.

The render pack Akar downloads is published under Releases (`pack-<version>`).

## Licences

akar-render is under the Apache License 2.0 (see `LICENSE`). The render pack also contains Cycles
and the libraries it is built with, each under its own licence; see `NOTICE.md` and the
`licenses/` folder inside the pack.
