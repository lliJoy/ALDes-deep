"""BOS, configurable-length graph tokens without consecutive repeats, EOS."""

import numpy as np


class StructureGrammar:
    def __init__(self, structure_count, max_structures=6, min_structures=1):
        if structure_count < 1 or min_structures < 1 or max_structures < min_structures:
            raise ValueError(
                "Structure count and length limits must satisfy "
                "1 <= min_structures <= max_structures."
            )
        if structure_count == 1 and min_structures > 1:
            raise ValueError(
                "At least two structures are required when the minimum length "
                "exceeds one and consecutive repeats are disabled."
            )
        self.structure_count = structure_count
        self.min_structures = min_structures
        self.max_structures = max_structures
        self.begin_index = structure_count
        self.end_index = structure_count + 1
        self.vocab_size = structure_count + 2

    def allowed_next_tokens(self, prefix):
        prefix = np.asarray(prefix).reshape(-1).tolist()
        if not prefix or prefix[0] != self.begin_index:
            raise ValueError("Structure program must begin with BOS.")
        values = prefix[1:]
        mask = np.zeros(self.vocab_size, dtype=bool)
        if self.end_index in values:
            end = values.index(self.end_index)
            if (
                not all(0 <= v < self.structure_count for v in values[:end])
                or not all(v == self.end_index for v in values[end:])
                or not self.min_structures <= end <= self.max_structures
            ):
                raise ValueError("Invalid structure program padding or length.")
            mask[self.end_index] = True
            return mask
        if (
            any(not 0 <= v < self.structure_count for v in values)
            or len(values) > self.max_structures
        ):
            raise ValueError("Unknown structure token or excessive length.")
        mask[: self.structure_count] = len(values) < self.max_structures
        if values and len(values) < self.max_structures:
            mask[values[-1]] = False
        mask[self.end_index] = len(values) >= self.min_structures
        return mask

    def decode(self, sequence):
        prefix = []
        raw = np.asarray(sequence).reshape(-1)
        if (
            raw.size < 3
            or not np.isfinite(raw).all()
            or not np.equal(raw, raw.astype(int)).all()
        ):
            raise ValueError("Expected a complete integer structure program.")
        values = raw.astype(int).tolist()
        if values[0] != self.begin_index:
            raise ValueError("Structure program must begin with BOS.")
        prefix = values[:1]
        for token in values[1:]:
            if (
                not 0 <= token < self.vocab_size
                or not self.allowed_next_tokens(prefix)[token]
            ):
                raise ValueError("Illegal structure program.")
            prefix.append(token)
        if self.end_index not in values:
            raise ValueError("Structure program must terminate with EOS.")
        return values[1 : values.index(self.end_index)]
