"""Independent outcome audit; imports no simulation/runner metric helpers."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np

ROOT = None
OUTPUT = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(count=len(a), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(np.max(a)),
                mean=float(np.mean(a)))


def main():
    rows = []
    discrepancy = []
    hashes = {}
    max_ep_difference = max_ef_difference = 0.
    summary = json.loads((ROOT / 'campaign_summary.json').read_text())
    for path in sorted((ROOT / 'traces_npz').glob('*/*.npz')):
        jpath = ROOT / 'episode_results' / path.parent.name / (path.stem + '.json')
        rec = json.loads(jpath.read_text())
        hashes[str(path.relative_to(ROOT))] = digest(path)
        hashes[str(jpath.relative_to(ROOT))] = digest(jpath)
        with np.load(path) as z:
            assert len(z['time_s']) == 220
            assert np.allclose(z['time_s'], np.arange(2, 441, 2), atol=1e-6)
            truth = np.column_stack([z['truth_' + c] for c in 'xyz'])
            estimate = np.column_stack([z['estimate_' + c] for c in 'xyz'])
            ep = np.linalg.norm(truth - estimate, axis=1)
            assert np.allclose(ep, z['localization_error_m'], equal_nan=True)
            max_ep_difference = max(max_ep_difference, float(np.nanmax(np.abs(ep-z['localization_error_m']))))
            # Constant-velocity truth trajectories reconstructed from the first
            # exact broadcast. Later online rows may occur one 0.1-s substep after
            # their nominal integer timestamp; do not substitute them for truth
            # at the action endpoint. The fixed desired offset is 120 m aft.
            assert np.allclose(z['online_t_s'][1::2], z['time_s'], atol=1e-6)
            assert np.allclose(z['online_leader_velocity_mps'],z['online_leader_velocity_mps'][0])
            centers = (z['online_leader_position_m'][0].mean(axis=0)
                       + z['online_leader_velocity_mps'][0].mean(axis=0)
                       * (z['time_s'][:,None]-z['online_t_s'][0]))
            headings = z['online_leader_velocity_mps'][1::2].mean(axis=1).copy()
            headings[:, 2] = 0.
            headings /= np.linalg.norm(headings, axis=1)[:, None]
            desired = centers - 120.*headings
            ef = np.linalg.norm(truth - desired, axis=1)
            assert np.allclose(ef, z['formation_error_truth_m'], atol=1e-7)
            max_ef_difference = max(max_ef_difference, float(np.max(np.abs(ef-z['formation_error_truth_m']))))
            ok = np.isfinite(ep) & np.isfinite(ef) & (ep < 7.) & (ef < 8.)
            track = z['phase_track'] > .5
            indices = np.flatnonzero(track)
            actions = np.column_stack([z['action_' + c] for c in ('speed','yaw','pitch')])
            triples = track[2:] & track[1:-1] & track[:-2]
            curvature = np.diff(actions, n=2, axis=0)[triples]
            changes = track & ~np.r_[False, track[:-1]]
            ep_start = np.r_[np.nan, ep[:-1]]
            row = dict(
                seed=int(rec['episode_seed']), controller=rec['controller_name'],
                terminal_joint_success=bool(ok[-1]),
                tail80_joint_success=bool(np.count_nonzero(ok[-50:]) >= 40),
                dwell15_joint_success=bool(np.all(ok[-15:])),
                tail50_joint_occupancy=float(np.mean(ok[-50:])),
                terminal_localization_error_m=float(ep[-1]),
                terminal_formation_error_m=float(ef[-1]),
                terminal_localization_success=bool(ep[-1]<7.),
                terminal_formation_success=bool(ef[-1]<8.),
                first_track_time_s=float(z['action_start_time_s'][indices[0]]) if len(indices) else None,
                post_track_saturation_fraction=float(np.mean(np.any(np.abs(actions[track])>=.98,axis=1))) if len(indices) else None,
                post_track_action_curvature_rms=float(np.sqrt(np.mean(np.sum(curvature**2,axis=1)))) if len(curvature) else None,
                unsafe_track_start_count=int(np.count_nonzero(track & (~np.isfinite(ep_start) | (ep_start>=7.)))),
                unsafe_track_end_count=int(np.count_nonzero(track & (~np.isfinite(ep) | (ep>=7.)))),
                unsafe_transition_count=int(np.count_nonzero(changes & (~np.isfinite(ep_start) | (ep_start>=7.)))),
                maximum_track_localization_error_m=float(np.max(ep[track])),
                minimum_formation_error_m=float(np.min(ef)),
                formation_error_at_track_start_m=float(ef[indices[0]-1]),
                localization_error_at_track_start_m=float(ep[indices[0]-1]),
                first_joint_success_time_s=float(z['time_s'][np.flatnonzero(ok)[0]]) if np.any(ok) else None,
                joint_success_total_endpoint_count=int(np.count_nonzero(ok)),
                track_segment_count=int(np.count_nonzero(changes)),
                formation_error_checkpoints_m={str(t):float(ef[int(t/2)-1]) for t in (100,180,240,300,340,380,400,420,440)},
                last50_formation_m=[float(x) for x in ef[-50:]],
                last50_localization_m=[float(x) for x in ep[-50:]],
            )
            for key in ('terminal_joint_success','tail80_joint_success','dwell15_joint_success','tail50_joint_occupancy','terminal_localization_error_m','terminal_formation_error_m'):
                if not np.isclose(row[key], rec[key], atol=1e-9): discrepancy.append([path.name,key,row[key],rec[key]])
            if not np.isclose(row['first_track_time_s'], rec['gate']['first_track_action_time_s']):
                discrepancy.append([path.name,'first_track'])
            for key,jsonkey in (('unsafe_track_start_count','false_locked_action_start_count'),('unsafe_track_end_count','false_locked_action_end_count'),('unsafe_transition_count','false_transition_count')):
                if row[key]!=rec['gate']['exact'][jsonkey]: discrepancy.append([path.name,key])
            # The nested legacy track diagnostic uses exact saturation, not the
            # paper/campaign threshold .98. Verify each definition separately.
            exact_saturation=float(np.mean(np.any(np.abs(actions[track]) >= 1.-1e-7,axis=1)))
            if not np.isclose(exact_saturation,rec['controller']['track_metrics']['any_channel_saturation_fraction']): discrepancy.append([path.name,'legacy_exact_saturation'])
            rows.append(row)
    assert len(rows)==100 and len({(r['seed'],r['controller']) for r in rows})==100
    groups = {}
    binary = ('terminal_joint_success','tail80_joint_success','dwell15_joint_success')
    metrics = ('terminal_localization_error_m','terminal_formation_error_m','first_track_time_s','post_track_saturation_fraction','post_track_action_curvature_rms')
    for controller in sorted({r['controller'] for r in rows}):
        rr = [r for r in rows if r['controller']==controller]
        g = dict(episodes=len(rr))
        for k in binary:
            g[k]=dict(count=sum(r[k] for r in rr),rate=float(np.mean([r[k] for r in rr])),failed_seeds=[r['seed'] for r in rr if not r[k]])
            summary_key=k.replace('_joint_success','_success_count')
            if g[k]['count']!=summary['metrics_by_controller'][controller][summary_key]: discrepancy.append([controller,k,'summary'])
        for k in metrics:
            g[k]=distribution([r[k] for r in rr])
            for stat in ('count','median','p95','max','mean'):
                if not np.isclose(g[k][stat],summary['metrics_by_controller'][controller][k][stat],atol=1e-9): discrepancy.append([controller,k,stat,g[k][stat],summary['metrics_by_controller'][controller][k][stat]])
        for k in ('unsafe_track_start_count','unsafe_track_end_count','unsafe_transition_count'):
            g[k]=sum(r[k] for r in rr)
        g['terminal_localization_success_count']=sum(r['terminal_localization_success'] for r in rr)
        g['terminal_formation_success_count']=sum(r['terminal_formation_success'] for r in rr)
        groups[controller]=g
    pairing={}
    for metric in binary:
        a={r['seed']:r[metric] for r in rows if r['controller']=='baseline_pid'}
        b={r['seed']:r[metric] for r in rows if r['controller']=='delay_aware'}
        pairing[metric]=dict(both_success=sum(a[s] and b[s] for s in a),baseline_only=sum(a[s] and not b[s] for s in a),delay_aware_only=sum(b[s] and not a[s] for s in a),neither=sum(not a[s] and not b[s] for s in a))
    report=dict(status='PASS' if not discrepancy else 'FAIL',recomputed_from='100 NPZ traces with coordinate-level position/formation-error reconstruction',discrepancies=discrepancy,max_absolute_error_reconstruction_difference_m=dict(localization=max_ep_difference,formation=max_ef_difference),notes=['Campaign/paper saturation uses abs(action)>=0.98. The nested historical controller.track_metrics diagnostic uses abs(action)>=1-1e-7; both definitions were verified separately and must not be mixed.','Constant leader truth positions are reconstructed from the first broadcast and exact action time; some later broadcast rows are sampled one 0.1-s integration substep after their nominal stored integer timestamp.'],metrics=groups,paired_binary=pairing,rows=rows,input_sha256=hashes)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','input_sha256')},indent=2))
    return int(bool(discrepancy))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    ROOT=args.campaign.resolve()
    OUTPUT=args.output.resolve()
    if OUTPUT == ROOT or ROOT in OUTPUT.parents:
        parser.error('Write the independent audit outside the frozen campaign')
    raise SystemExit(main())
