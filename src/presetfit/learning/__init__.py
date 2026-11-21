"""Machine learning for PresetFit: a neural network that predicts the sliders.

The model looks at the photo and the reference and predicts all slider values
at once. It is trained by rendering its own guess with a differentiable copy of
the edit pipeline (:mod:`presetfit.learning.render`) and comparing the result
with the correctly edited photo, so no hand-made slider labels are needed.

Everything here needs the optional training dependencies::

    pip install -e ".[train]"

The rest of PresetFit never imports this package, so the app keeps working
without PyTorch installed.
"""
