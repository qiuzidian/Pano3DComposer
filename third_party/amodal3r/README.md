# Amodal3R (vendored)

Copy of the `amodal3r` package from https://github.com/Sm0kyWu/Amodal3R (S-Lab License 1.0,
see `LICENSE.txt`), with the following changes for Pano3DComposer:

* `renderers/gaussian_render.py`: uses the depth/alpha-returning rasterizer
  (https://github.com/ashawkey/diff-gaussian-rasterization) and returns the rendered depth.
* `utils/render_utils.py`: `render_snapshot` renders the four reference views fed to the
  object-to-world predictor (yaw offset -16 deg, pitch 20 deg, radius 1.5, fov 50, white
  background) and returns cameras, colors and depths.
* `utils/postprocessing_utils.py`: `to_glb` keeps the z-up frame of the Gaussians.
* `models/sparse_structure_flow*.py`, `models/structured_latent_vae/decoder_*.py`: dtype
  casts so the pipeline can run in fp16.
* `pipelines/image_to_3d.py`: sampling noise and latent normalization follow the model dtype (fp16).
