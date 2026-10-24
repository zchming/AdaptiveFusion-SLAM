#!/usr/bin/env python3
"""Gate cached primary 3D-2D updates before reading recorded truth evaluation.

No new fitting, trajectory updates, edge admission or online tracking.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from adaptivefusion.frontend_features import read_features
from adaptivefusion.geometry import unproject
from adaptivefusion.pair_update_guard import CONFIG, guard_update
from adaptivefusion.state import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['comparison', 'matching', 'features', 'evaluation', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output exists; refuse overwrite')
    cache = json.loads(args.comparison.read_text())
    if cache['truth_read'] is not False: raise ValueError('fit-only source required')
    decisions = []
    hashes = {'comparison': file_sha256(args.comparison)}
    for case in cache['cases']:
        path = args.matching/'descriptor'/f"{case['pair']}.json"
        digest = file_sha256(path)
        if digest != cache['input_sha256']['proposal/'+path.name]: raise ValueError('proposal hash mismatch')
        hashes['proposal/'+path.name] = digest
        proposal = json.loads(path.read_text())
        a, b = proposal['source_keyframe'], proposal['target_keyframe']
        features = []
        for kid in [a, b]:
            fp = args.features/f'{kid}.features'; digest = file_sha256(fp)
            if digest != cache['input_sha256']['feature/'+fp.name]: raise ValueError('feature hash mismatch')
            f = read_features(fp)
            if f.metadata['keyframe_id'] != kid: raise ValueError('feature identity mismatch')
            features.append(f); hashes['feature/'+fp.name] = digest
        fa, fb = features
        if fa.metadata['camera'] != fb.metadata['camera'] or fa.metadata['depth'] != fb.metadata['depth']:
            raise ValueError('calibration mismatch')
        ids, source, pixels = [], [], []
        for row in proposal['records']:
            i, j = row['source_feature'], row['target_feature']
            ids.append(i)
            if i % 2 == 0: continue
            z = int(fa.raw_depth[i])/fa.metadata['depth'][0]
            source.append(unproject(fa.metadata['camera'], fa.keypoints[i, :2], z))
            pixels.append(fb.keypoints[j, :2])
        if ([i for i in ids if i % 2] != case['validation_source_ids']
                or [i for i in ids if i % 2 == 0] != case['fit_source_ids']):
            raise ValueError('split identity mismatch')
        arm = next(a for a in case['arms'] if a['model'] == '3d2d' and a['initialization'] == 'raw_baseline')
        decision = guard_update(np.array(case['raw_baseline']), np.array(arm['fit']['t_target_source']),
                                np.array(source).reshape(-1, 3), np.array(pixels).reshape(-1, 2),
                                fa.metadata['camera'], solver_applied=arm['fit']['applied'])
        decisions.append(dict(pair=case['pair'], solver_applied=arm['fit']['applied'],
                              check_source_ids=case['validation_source_ids'], decision=decision))
    args.output.mkdir(parents=True)
    pre = args.output/'guard-only.json'
    pre.write_text(json.dumps(dict(config=CONFIG, cases=decisions, input_sha256=hashes,
                                  truth_read=False, scope='odd pixels are decision inputs; no longer independent validation'), indent=2)+'\n')
    pre_hash = file_sha256(pre)
    # Recorded GT is used only after every selection is finalized.
    evaluation = json.loads(args.evaluation.read_text())
    if evaluation['fit_only_sha256'] != hashes['comparison']: raise ValueError('recorded evaluation provenance mismatch')
    contexts = {c['pair']: c for c in evaluation['cases']}
    for case in decisions:
        context = contexts[case['pair']]
        candidate = next(a for a in context['arms'] if a['model'] == '3d2d' and a['initialization'] == 'raw_baseline')
        chosen = candidate['fit']['t_target_source'] if case['decision']['accepted'] else context['raw_baseline']
        if chosen != case['decision']['selected_t_target_source']: raise ValueError('evaluation model mismatch')
        case['selected_truth_errors'] = candidate['truth_errors'] if case['decision']['accepted'] else context['raw_baseline_truth']
        case['unguarded_truth_errors'] = candidate['truth_errors']
        case['raw_baseline_truth_errors'] = context['raw_baseline_truth']
    assert file_sha256(pre) == pre_hash
    (args.output/'report.json').write_text(json.dumps(dict(cases=decisions, config=CONFIG,
        guard_only_sha256=pre_hash, recorded_evaluation_sha256=file_sha256(args.evaluation),
        scope='cached development update decisions only; no fresh independence or causal tracking benefit'), indent=2)+'\n')
    print(json.dumps({'cases': len(decisions), 'accepted': sum(c['decision']['accepted'] for c in decisions)}))


if __name__ == '__main__': main()
