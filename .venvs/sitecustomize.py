import sys

_PATCHED = False


def _patch_mamba_transformers_generation() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    import transformers.generation as generation
    if hasattr(generation, 'GreedySearchDecoderOnlyOutput'):
        return
    from transformers.generation.utils import GenerateDecoderOnlyOutput
    generation.GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
    generation.SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput


class _MambaSsmPatchFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mamba_ssm' or fullname.startswith('mamba_ssm.'):
            _patch_mamba_transformers_generation()
        return None


sys.meta_path.insert(0, _MambaSsmPatchFinder())
