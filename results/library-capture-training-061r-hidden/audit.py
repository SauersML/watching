"""Recompute paired endpoint metrics and exact binomial intervals from raw rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta


def audit(path):
    artifact = json.loads(path.read_text())
    rows = artifact['rows']
    pairs = {}
    for row in rows:
        pairs.setdefault(row['index'], {})[bool(row['capture'])] = row
    assert len(rows) == 2 * len(pairs)
    assert all(set(pair) == {False, True} for pair in pairs.values())
    assert all(pair[False]['context_id'] == pair[True]['context_id'] for pair in pairs.values())
    successful = [all(row['top_token_id'] == row['target_token_id'] for row in pair.values()) for pair in pairs.values()]
    n, k = len(successful), sum(successful)
    interval = [0 if k == 0 else float(beta.ppf(.025, k, n-k+1)),
                1 if k == n else float(beta.ppf(.975, k+1, n-k))]
    shifts = np.array([pair[True]['yes_minus_no'] - pair[False]['yes_minus_no'] for pair in pairs.values()])
    rng = np.random.default_rng(20261017)
    sampled = shifts[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    ce = float(np.mean([row['ce'] for row in rows]))
    assert abs(k/n-artifact['summary']['greedy_both_conditions_correct']) < 1e-12
    assert abs(ce-artifact['summary']['mean_ce']) < 1e-6
    ce_error = max(abs(row['ce'] + row['logp_positive' if row['capture'] else 'logp_negative']) for row in rows)
    assert ce_error < 1e-6
    assert all(row['greedy_correct'] == (row['top_token_id'] == row['target_token_id']) for row in rows)
    return {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'n_prompt_pairs': n, 'both_correct': k, 'fraction': k/n,
            'exact_clopper_pearson_ci95': interval, 'mean_ce': ce,
            'paired_log_odds_shift': float(shifts.mean()),
            'prompt_pair_bootstrap_shift_ci95': np.quantile(sampled, [.025, .975]).tolist(),
            'bootstrap_draws': 10000, 'bootstrap_seed': 20261017,
            'positive_shifts': int((shifts > 0).sum()),
            'wrong_answers': sum(not row['greedy_correct'] for row in rows),
            'wrong_answers_with_yes_or_no_probability_at_least_0_9': sum(not row['greedy_correct'] and max(math.exp(row['logp_positive']), math.exp(row['logp_negative'])) >= .9 for row in rows),
            'max_ce_reconstruction_error': ce_error,
            'full_vocab_logits_available': False,
            'verification_scope': 'Stored top-token IDs and full-vocabulary-normalized target log probabilities; original full logits are not stored'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fit', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    paths = [args.fit/f'{split}_1024_{runtime}.json' for runtime in ['split_inference', 'wholegraph_transfer'] for split in ['train', 'validation', 'test']]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f'Final artifacts missing: {missing}')
    args.output.write_text(json.dumps({'endpoints': [audit(path) for path in paths],
        'interval_scope': 'Pointwise intervals on complete prompt pairs; historical test is exploratory. Binomial intervals do not model family dependence or training-seed/compiler uncertainty.'}, indent=2)+'\n')
