"""Deterministic synthetic needle-in-a-haystack data for retrieval experiments."""
import numpy as np
import torch

# These are token IDs, deliberately independent of the TinyStories tokenizer.
# Every sequence ends in [QUERY, needle_key, ANSWER], and the target is needle_value.
NEEDLE, DISTRACTOR, QUERY, ANSWER, SEP = range(1, 6)
KEY_START, KEY_COUNT = 8, 120
VALUE_START, VALUE_COUNT = 128, 128


def _positions(rng, batch_size, num_distractors, depth):
    if num_distractors == 0:
        return np.zeros(batch_size, dtype=np.int64)
    slots = num_distractors + 1
    if depth == "early":
        lo, hi = 0, max(1, slots // 3)
    elif depth == "middle":
        lo, hi = slots // 3, max(slots // 3 + 1, 2 * slots // 3)
    elif depth == "late":
        lo, hi = max(2 * slots // 3, 0), slots
    else:
        lo, hi = 0, slots
    return rng.integers(lo, hi, size=batch_size, endpoint=False)


def make_batch(batch_size, num_distractors, depth, seed, device):
    """Return fixed-length prompts and answer-only targets.

    Distractor keys never match the queried key. Needle position is stratified by
    its fact slot (early/middle/late), while all fact values are independently
    random, preventing shortcuts from value or position.
    """
    rng = np.random.default_rng(seed)
    needle_keys = rng.integers(KEY_START, KEY_START + KEY_COUNT, size=batch_size)
    needle_values = rng.integers(VALUE_START, VALUE_START + VALUE_COUNT, size=batch_size)
    slots = num_distractors + 1
    facts = np.empty((batch_size, slots, 4), dtype=np.int64)
    facts[:, :, 0] = DISTRACTOR
    facts[:, :, 1] = rng.integers(KEY_START, KEY_START + KEY_COUNT, size=(batch_size, slots))
    facts[:, :, 2] = rng.integers(VALUE_START, VALUE_START + VALUE_COUNT, size=(batch_size, slots))
    facts[:, :, 3] = SEP
    needle_slot = _positions(rng, batch_size, num_distractors, depth)
    rows = np.arange(batch_size)
    # Ensure a distractor cannot satisfy the query; duplicates among distractors are harmless.
    collision = facts[:, :, 1] == needle_keys[:, None]
    facts[:, :, 1][collision] = KEY_START + (facts[:, :, 1][collision] - KEY_START + 1) % KEY_COUNT
    facts[rows, needle_slot, 0] = NEEDLE
    facts[rows, needle_slot, 1] = needle_keys
    facts[rows, needle_slot, 2] = needle_values
    query = np.stack((np.full(batch_size, QUERY), needle_keys, np.full(batch_size, ANSWER)), axis=1)
    prompts = np.concatenate((facts.reshape(batch_size, -1), query), axis=1)
    return torch.from_numpy(prompts).to(device), torch.from_numpy(needle_values).to(device)
