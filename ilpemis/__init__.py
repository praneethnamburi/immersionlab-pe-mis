"""immersionlab-pe-mis: preprocessing + teaching materials for the MIT
Professional Education course *Mastering Integrated Systems*.

The build-side submodules (``preprocess`` / ``analysis`` / ``atem`` / ``bundle.build_bundle``)
import lab tooling (delsys, immersionlab, mithic, projects.wobble) and run in the
**b4** env -- import them explicitly. ``bundle.load_bundle`` is dependency-light
(h5py only) so the Colab teaching layer can read a bundle without the lab stack.
"""
__all__ = ["preprocess", "analysis", "atem", "bundle"]
