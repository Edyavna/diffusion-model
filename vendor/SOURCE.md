# NVIDIA EDM

Source: https://github.com/NVlabs/edm

Revision: `008a4e5316c8e3bfe61a62f874bddba254295afb`.

`edm_networks.py` is `training/networks.py` from that revision. The only code
changes are removal of the `torch_utils.persistence` import and all
`@persistence.persistent_class` decorators. Layer definitions and forward /
backward computations are unchanged. Checkpoints here store state_dict tensors
and model configuration instead of pickled model objects.

Copyright NVIDIA CORPORATION & AFFILIATES. The upstream code is licensed under
CC BY-NC-SA 4.0; see the included `EDM_LICENSE.txt`. Those license conditions apply
to this vendored code; this is not an unrestricted commercial-use dependency.
