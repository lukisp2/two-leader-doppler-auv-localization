#!/usr/bin/env python3
"""Export the complete difficult-start campaign with a narrow privacy boundary.

Only selection audit root labels, the frozen glue runner's private lookup root,
and the two dependent contract hashes change. Every raw result, trace, noise
realization, scientific source and independently calculated metric is preserved.
The original and projected hashes are retained in PROJECTION_MANIFEST.json.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

PRIVATE = re.compile(r'/(?:Users|home)/[^\s"\']+')
FIXED_FILES = ('selected_scenarios.json', 'campaign_summary.json', 'pairing_checks.json',
               'episode_arm_summary.csv', 'control/campaign_contract.json',
               'analysis/raw_outcome_audit.json', 'analysis/independent_integrity_audit.json')
DIRECTORIES = {'episode_results':100, 'traces_npz':100, 'noise_tapes':50,
               'control/source_snapshot':27}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def public_root_label(value):
    for marker, label in (('/release_repo_staging/', 'repository/'),
                          ('/doi_archive_staging/', 'archive/'),
                          ('/UUV_ART2/', 'historical_source/')):
        if marker in value:
            return label + value.split(marker,1)[1]
    raise ValueError('Unexpected private audit root; explicit review required')

def project(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists() or campaign == output or campaign in output.parents:
        raise ValueError('Use a new export directory outside the immutable campaign')
    paths = [campaign/p for p in FIXED_FILES]
    for directory, count in DIRECTORIES.items():
        found = sorted(p for p in (campaign/directory).rglob('*') if p.is_file())
        if len(found) != count or any(p.is_symlink() for p in found):
            raise ValueError(f'Unexpected cardinality or symlink: {directory}')
        paths += found
    if not all(p.is_file() and not p.is_symlink() for p in paths):
        raise ValueError('Missing required audit or campaign file')
    old_contract = json.loads((campaign/'control/campaign_contract.json').read_text())
    selection = json.loads((campaign/'selected_scenarios.json').read_text())
    if sha(campaign/'selected_scenarios.json') != old_contract['selection_manifest_sha256']:
        raise ValueError('Frozen selection hash mismatch')
    if len(selection['selected_seeds']) != 50 or selection['selected_seeds'] != old_contract['seeds']:
        raise ValueError('Frozen scenario set mismatch')
    for name, expected in old_contract['source_sha256'].items():
        if sha(campaign/'control/source_snapshot'/name) != expected:
            raise ValueError(f'Frozen source hash mismatch: {name}')
    output.mkdir(parents=True)
    for source in paths:
        dest = output/source.relative_to(campaign)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    selection['freshness_audit']['scope_roots'] = [public_root_label(v) for v in selection['freshness_audit']['scope_roots']]
    (output/'selected_scenarios.json').write_text(json.dumps(selection,indent=2,sort_keys=True)+'\n')
    glue = output/'control/source_snapshot/revision_runner.py'
    text = glue.read_text()
    text, n = re.subn(r'old = Path\("/Users/[^"\n]+/UUV_ART2"\)',
                      'old = Path("historical_source")', text)
    if n != 1:
        raise ValueError('Expected exactly one private lookup root in frozen glue')
    glue.write_text(text)
    contract = json.loads(json.dumps(old_contract))
    contract['source_sha256']['revision_runner.py'] = sha(glue)
    contract['selection_manifest_sha256'] = sha(output/'selected_scenarios.json')
    (output/'control/campaign_contract.json').write_text(json.dumps(contract,indent=2,sort_keys=True)+'\n')
    changed = {'selected_scenarios.json','control/source_snapshot/revision_runner.py','control/campaign_contract.json'}
    entries = []
    for source in sorted(paths):
        relative = source.relative_to(campaign).as_posix()
        dest = output/relative
        old, new = sha(source), sha(dest)
        if relative not in changed and old != new:
            raise ValueError(f'Unexpected changed artifact: {relative}')
        if dest.suffix in {'.json','.py','.csv','.txt'} and PRIVATE.search(dest.read_text()):
            raise ValueError(f'Private absolute path remains: {relative}')
        entries.append({'path':relative,'original_sha256':old,'public_sha256':new,
                        'changed':old != new})
    manifest = {'schema_version':1,'scientific_values_changed':False,
                'original_contract_sha256':sha(campaign/'control/campaign_contract.json'),
                'source_file_count':len(paths), 'cardinalities':DIRECTORIES,
                'transformations':[
                    'selection freshness scope roots replaced by repository/archive/historical_source aliases',
                    'frozen runner private lookup root replaced by historical_source; use the portable public runner for execution',
                    'contract source and selection hashes updated to those projected files'],
                'files':entries}
    (output/'PROJECTION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=project(args.campaign,args.output)
    print(f"PASS: {result['source_file_count']} files projected; 250 raw numerical/result files unchanged")
