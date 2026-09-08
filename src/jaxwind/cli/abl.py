"""Small argument adapter for the benchmark convenience scripts."""
from __future__ import annotations
import argparse
from pathlib import Path
import json


def parser(default_config=None):
    result = argparse.ArgumentParser()
    result.add_argument("config", type=Path, nargs="?" if default_config else None, default=default_config)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--output", type=Path)
    result.add_argument("--max-steps", type=int)
    return result


def run(arguments):
    if arguments.dry_run:
        from jaxwind.config.validation import check_case
        result = check_case(arguments.config)
        print(json.dumps(result, indent=2))
        return result
    from jaxwind.runtime.engine import run as execute
    return execute(arguments.config, output=arguments.output, max_steps=arguments.max_steps).summary
