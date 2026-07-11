"""Pure-Python helpers for building language-instruction candidate batches.

Kept dependency-free (no torch/tensorflow/simpler_env) so this logic can be
unit tested in isolation from the rest of the eval pipeline.
"""

from typing import List, Optional


def build_unique_prompts(
    task_description: str,
    rephrased_list: Optional[List[str]],
    lang_rephrase_num: int,
) -> List[str]:
    """Build the list of distinct instruction candidates for one policy batch.

    `task_description` always occupies the first slot. The remaining slots are
    filled from `rephrased_list`, skipping any entry equal to `task_description`
    so a candidate is never sent twice in the same batch (this can otherwise
    happen once the verifier reassigns `task_description` to a phrase that is
    itself one of the rephrases).
    """
    if rephrased_list is None or lang_rephrase_num <= 1:
        return [task_description]

    remaining = [p for p in rephrased_list if p != task_description]
    return [task_description] + remaining[: lang_rephrase_num - 1]
