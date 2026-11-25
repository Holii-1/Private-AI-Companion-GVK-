# GVK Private AI - Hardware Dock (`GVK-DOCK MK.I`)

> [!NOTE]
> **Status: Unfinished Concept / Planned for Rework.**
> The physical `GVK-DOCK MK.I` hardware console has **not** been built or implemented yet. The files in `hardware/` are an early 3D visual prototype based on the original optical-disc concept and will be redesigned—especially following the addition of offline USB drive support in `v1.1.1 - Optical Funeral`.

---

## What Exists Today (`hardware/`)

| File | Description |
| :--- | :--- |
| [generate_3d_dock.py](../hardware/generate_3d_dock.py) | Procedural Python CAD generator that builds the slanted console mesh and exports `.obj`, `.mtl`, binary `.stl`, and the interactive HTML viewer. |
| [gvk_dock_mk1.obj](../hardware/gvk_dock_mk1.obj) / [.mtl](../hardware/gvk_dock_mk1.mtl) | Colored 3D prototype assembly viewable in Blender or macOS QuickLook. |
| [gvk_dock_viewer.html](../hardware/gvk_dock_viewer.html) | Standalone interactive Three.js browser viewer with simulated cartridge eject and kill-switch toggle. |

### Regenerating the 3D Prototype Files
```bash
python3 hardware/generate_3d_dock.py
```
This regenerates `gvk_dock_mk1.obj`, `gvk_dock_mk1.mtl`, a slicer-ready `gvk_dock_mk1.stl`, and `gvk_dock_viewer.html` inside `hardware/`.

---

## Original Concept & Planned Rework

The original MK.I concept envisioned a desktop enclosure housing:
* A vertical slim optical drive bay + 3 collector cartridge slots (`friend_1`, `friend_2`, `friend_3`).
* Status LEDs and a physical aviation-style **Kill-Switch** toggle wired to cut USB bus power to the media reader, triggering the 3-second hardware Kill-Switch daemon in `core/crypto_vault.py` to terminate `llama-server` and zero-wipe RAM.

Because `v1.1.1 - Optical Funeral` expanded physical delivery to fully offline USB drives alongside DVDs, the hardware form factor, bay layout, and kill-switch wiring will be reworked in a future revision before physical fabrication.
