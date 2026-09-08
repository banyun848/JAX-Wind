"""Installed JAX-Wind commands; all execution delegates to package services."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parser():
    result = argparse.ArgumentParser(prog="jaxwind")
    commands = result.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="resolve and validate without constructing simulation fields")
    check.add_argument("case", type=Path)
    run = commands.add_parser("run", help="start a simulation in an empty output directory")
    run.add_argument("case", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--max-steps", type=int)
    resume = commands.add_parser("resume", help="continue a new-format run")
    resume.add_argument("directory", type=Path)
    resume.add_argument("--max-steps", type=int)
    workflow = commands.add_parser("workflow", help="execute declared stage dependencies")
    workflow.add_argument("case", type=Path)
    workflow.add_argument("--stage")
    workflow.add_argument("--resume", action="store_true")
    workflow.add_argument("--output", type=Path)
    workflow.add_argument("--max-steps", type=int)
    case = commands.add_parser("case").add_subparsers(dest="case_command", required=True)
    derive = case.add_parser("derive", help="write an inherited case with independent output")
    derive.add_argument("base", type=Path)
    derive.add_argument("--output", type=Path, required=True)
    derive.add_argument("--cells", "--resolution", nargs=3, type=int)
    derive.add_argument("--cfl", "--cfl-ceiling", "--cfl-threshold", type=float)
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "case":
            from jaxwind.config.document import derive_case
            print(derive_case(arguments.base, arguments.output, cells=arguments.cells, cfl=arguments.cfl))
        elif arguments.command == "check":
            from jaxwind.config.document import tomllib
            with arguments.case.open("rb") as stream:
                document = tomllib.load(stream)
            if "stages" in document:
                from jaxwind.workflows.engine import check_workflow
                report = check_workflow(arguments.case)
            else:
                from jaxwind.config.validation import check_case
                report = check_case(arguments.case)
            print(json.dumps(report, indent=2))
        elif arguments.command == "run":
            from jaxwind.runtime.engine import run
            result = run(arguments.case, output=arguments.output, max_steps=arguments.max_steps)
            print(json.dumps(result.summary, indent=2))
        elif arguments.command == "resume":
            from jaxwind.runtime.engine import resume
            print(json.dumps(resume(arguments.directory, max_steps=arguments.max_steps).summary, indent=2))
        else:
            from jaxwind.workflows.engine import execute
            print(json.dumps(execute(arguments.case, stage=arguments.stage, resume=arguments.resume, output=arguments.output, max_steps=arguments.max_steps), indent=2))
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
    return 0
