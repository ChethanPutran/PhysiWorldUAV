"""One entry point for all the live pages and plots of this project: ``python -m physiworld_uav.visualization <command> ...``.

Commands (details and examples in README.md next to this file):

    generation   serve the live dataset report (running / queued / done per episode, resources) while a dataset is generated
    report       write the dataset report once as a static HTML file
    progress     print generation progress and an ETA in the terminal
    training     live page with charts for a training sweep (writer + web server in one process)
    plan         distribution plots of a planned or generated dataset (from its manifest)
    wind         distribution plots of the wind actually stored in a dataset (needs the HDF5 shards)

Every command passes its remaining arguments to the module that implements it; ``<command> --help`` shows them.
"""
from __future__ import annotations

import sys

COMMANDS = {
    "generation": ("physiworld_uav.visualization.dataset_report", ["--serve"]),
    "report": ("physiworld_uav.visualization.dataset_report", []),
    "progress": ("physiworld_uav.visualization.gen_progress", []),
    "training": ("physiworld_uav.visualization.live_training", ["--serve"]),
    "plan": ("physiworld_uav.visualization.plot_plan_distribution", []),
    "wind": ("physiworld_uav.visualization.plot_wind_distribution", []),
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print(__doc__)
        if argv and argv[0] not in ("-h", "--help"):
            print(f"unknown command {argv[0]!r}", file=sys.stderr)
            sys.exit(2)
        return
    module, extra = COMMANDS[argv[0]]
    import importlib
    mod = importlib.import_module(module)
    mod.main(extra + argv[1:])


if __name__ == "__main__":
    main()
