"""Fixed stdlib guardian for one FEDOT candidate process group.

The dispatcher passes this helper an exact PGID and a private control pipe. If
the dispatcher exits before sending ``complete\n``, the helper terminates only
that candidate group. It performs no process discovery and executes no command.
"""

from __future__ import annotations

import argparse
import os
import select
import signal
import sys


def terminate_owned_group(candidate_pgid: int) -> int:
    try:
        os.killpg(candidate_pgid, signal.SIGKILL)
    except ProcessLookupError:
        return 0
    except OSError as exc:
        print(
            f"candidate_group_cleanup_failed pgid={candidate_pgid} "
            f"error={type(exc).__name__}:{exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pgid", required=True, type=int)
    parser.add_argument("--control-fd", required=True, type=int)
    parser.add_argument("--ready-fd", required=True, type=int)
    args = parser.parse_args()
    if args.candidate_pgid <= 1 or args.candidate_pgid == os.getpgrp():
        parser.error("candidate PGID must identify a separate, owned session")
    try:
        os.write(args.ready_fd, b"R")
    finally:
        os.close(args.ready_fd)
    try:
        readable, _, _ = select.select([args.control_fd], [], [])
        if not readable:
            return terminate_owned_group(args.candidate_pgid)
        message = os.read(args.control_fd, 64)
        if message == b"complete\n":
            return 0
        return terminate_owned_group(args.candidate_pgid)
    finally:
        os.close(args.control_fd)


if __name__ == "__main__":
    raise SystemExit(main())
