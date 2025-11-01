import random


class PersistentLanguageMapper:
    def __init__(self, mapping_candidates: dict[str, list[str]], seed: int = 42):
        """
        mapping_candidates: Dict mapping keys to list of possible output strings.
        Example: {'A': ['apple', 'apricot'], 'B': ['banana', 'blueberry']}
        """
        self.mapping_candidates = mapping_candidates
        self.mapping = {}
        self._random = random.Random(seed)

    def map(self, key: str) -> str:
        if key not in self.mapping:
            if key not in self.mapping_candidates:
                raise KeyError(f"No candidates defined for key '{key}'")
            self.mapping[key] = self._random.choice(self.mapping_candidates[key])
        return self.mapping[key]

    def reset(self):
        self.mapping.clear()  # keep RNG state to continue sequence
