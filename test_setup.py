#!/usr/bin/env python3
"""Check the LoVRR environment and model configuration."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--skip-models', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).parent
    config = yaml.safe_load(Path(args.config).read_text())
    errors = []
    for line in (root / 'requirements.txt').read_text().splitlines():
        requirement = Requirement(line)
        try:
            installed = Version(version(requirement.name))
            if installed not in requirement.specifier:
                errors.append(f'{requirement}: found {installed}')
        except PackageNotFoundError:
            errors.append(f'{requirement}: missing')
    if not args.skip_models:
        model = config['model']
        checkpoint = Path(model['model_path'])
        config_path = checkpoint / 'config.json'
        if not config_path.is_file():
            errors.append(f'Model config missing: {config_path}')
        else:
            model_config = json.loads(config_path.read_text())
            if model_config.get('model_type') not in ('qwen2_vl', 'agent_qwen2_vl'):
                errors.append('A Qwen2-VL checkpoint is required')
        base_path = Path(model['base_model_path'])
        if not (base_path / 'config.json').is_file():
            errors.append(f'Base model config missing: {base_path}')
        for role in ('planner', 'grounder', 'refiner', 'deepthinker'):
            path = Path(model.get('adapters', {}).get(role) or checkpoint / role)
            if not (path / 'adapter_config.json').is_file():
                errors.append(f'{role} adapter missing: {path}')
    for error in errors:
        print(error)
    if errors:
        return 1
    print('LoVRR setup checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
