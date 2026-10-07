"""adversarial-prompt-synthesizer: deterministic attack-variant multiplier.

Public surface:
    synth.transforms   deterministic transform library
    synth.generator    Synthesizer: seed -> variant corpus with lineage replay
    synth.novelty      novelty metrics vs seed + corpus pairwise distance
    synth.coverage     category x transform-family coverage matrix
    synth.corpus       JSONL corpus export / import
    synth.cli          command-line interface
"""

from . import transforms, generator, novelty, coverage, corpus, cli  # noqa: F401

__version__ = "0.1.0"
