"""Content-pinned stage reuse; retain stale outputs instead of mixing checkpoint identities.

Future-package helper only. prepare archives only explicitly listed outputs and
this stage's marker/log when dependency bytes change. commit requires dependencies
still match the pre-command pin and every expected output to exist.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def signature(inputs):
    hashes = {}
    for name, path in inputs.items():
        with Path(path).open('rb') as handle:
            hashes[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {'signature': digest, 'dependency_sha256': hashes}


def paths(root, stage):
    if not stage or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in stage):
        raise ValueError('Invalid stage name')
    return root / f'pin.{stage}.json', root / f'pending.{stage}.json', root / f'done.{stage}'


def prepare(root, stage, inputs, outputs, command=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    record, pending, marker = paths(root, stage)
    outputs = [Path(path).absolute() for path in outputs]
    if not outputs or len(outputs) != len(set(outputs)):
        raise ValueError('Expected unique, explicit stage outputs')
    for output in outputs:
        for dependency in inputs.values():
            if Path(dependency).resolve().is_relative_to(output.resolve()):
                raise ValueError('An output contains a dependency; refusing to archive it')
    expected = {**signature(inputs), 'outputs': [str(path) for path in outputs], 'command': list(command or [])}
    if marker.exists() and record.exists():
        previous = json.loads(record.read_text())
        if previous == expected and all(path.exists() for path in outputs):
            return 'current'
    stale = [path for path in [*outputs, marker, record, pending, root / f'{stage}.log']
             if path.exists() or path.is_symlink()]
    if stale:
        archive_root = root / 'stale-stages' / stage
        archive_root.mkdir(parents=True, exist_ok=True)
        index = 0
        while (archive_root / f'{index:04d}').exists():
            index += 1
        archive = archive_root / f'{index:04d}'
        archive.mkdir()
        moved = []
        for index, path in enumerate(stale):
            if path.is_relative_to(root):
                destination = archive / f'{index:02d}-{path.name}'
            else:
                # Large exported models must stay outside synchronized run outputs.
                external = path.parent / '.stale-stages'
                external.mkdir(exist_ok=True)
                identity = hashlib.sha256(str(root).encode()).hexdigest()[:12]
                destination = external / f'{stage}-{identity}-{archive.name}-{index:02d}-{path.name}'
                if destination.exists():
                    raise FileExistsError('External archive destination already exists')
            shutil.move(str(path), str(destination))
            moved.append({'original': str(path), 'archived': str(destination)})
        atomic_json(archive / 'archive.json', {'files': moved, 'replacement_pin': expected})
    atomic_json(pending, expected)
    return 'run'


def commit(root, stage, inputs, outputs, command=None):
    root = Path(root).resolve()
    record, pending, marker = paths(root, stage)
    output_paths = [str(Path(path).absolute()) for path in outputs]
    expected = {**signature(inputs), 'outputs': output_paths, 'command': list(command or [])}
    prepared = json.loads(pending.read_text())
    if prepared != expected:
        raise ValueError('Dependencies or output contract changed while stage ran')
    if not all(Path(path).exists() for path in output_paths):
        raise ValueError('Stage output is missing')
    atomic_json(record, expected)
    marker.touch()
    pending.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'commit'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--command-json', default='[]')
    parser.add_argument('--input', action='append', default=[])
    parser.add_argument('--output', action='append', type=Path, default=[])
    args = parser.parse_args()
    inputs = {}
    for item in args.input:
        name, path = item.split('=', 1)
        if name in inputs:
            parser.error('Duplicate dependency name')
        inputs[name] = Path(path)
    if not inputs:
        parser.error('At least one dependency is required')
    command = json.loads(args.command_json)
    if not isinstance(command, list) or any(not isinstance(arg, str) for arg in command):
        parser.error('Expected a JSON array of command arguments')
    if args.action == 'prepare':
        print(prepare(args.root, args.stage, inputs, args.output, command))
    else:
        commit(args.root, args.stage, inputs, args.output, command)


if __name__ == '__main__':
    main()
