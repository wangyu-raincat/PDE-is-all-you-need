"""Export a case's Nexent import files or an authenticated decision trace."""
import argparse
import json
from pathlib import Path

from competition_agent.service import Lab
from . import get_trace
from .retrieval import export_knowledge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--case-id")
    mode.add_argument("--trace-id")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lab = Lab(args.directory)
    if args.trace_id:
        trace = get_trace(lab,args.trace_id)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(trace,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    else:
        with lab.case(args.case_id,readonly=True) as state:
            lab.verify_seals(state)
            export_knowledge(state,args.output)


if __name__=="__main__":
    main()
