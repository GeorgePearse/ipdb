# Copyright (c) 2007-2016 Godefroid Chapelle and ipdb development team
#
# This file is part of ipdb.
# Redistributable under the revised BSD license
# https://opensource.org/licenses/BSD-3-Clause

"""Torch-aware debugging helpers.

A Python debugger only sees the host side of a PyTorch program. Kernel
launches are asynchronous, so an error raised by the device surfaces at
whatever line the host happened to reach, and a breakpoint tells you nothing
about what the GPU is doing. Two things recover most of that gap, and both
have to be arranged *before* the failure rather than after it:

``CUDA_LAUNCH_BLOCKING=1``
    Makes launches synchronous so a device-side error is raised at the call
    that caused it. The CUDA runtime reads this when it creates a context, so
    setting it after CUDA is initialised does nothing at all. Prefer the
    launcher (``python -m ipdb.gpu your_script.py``), which sets it before
    anything imports torch.

Memory history
    ``torch.cuda.memory._record_memory_history()`` records allocation stacks;
    dumping a snapshot on the way into the debugger turns an OOM from "it ran
    out" into an allocation trace you can open in the PyTorch memory viewer at
    https://pytorch.org/memory_viz.

Nothing here imports torch at module import time, so ``import ipdb.gpu`` is
safe in an environment without it, and the launcher can set the environment
before torch is ever loaded.
"""

from __future__ import print_function

import datetime
import os
import runpy
import sys

__all__ = [
    "enable",
    "enable_launch_blocking",
    "start_recording_memory",
    "dump_snapshot",
    "install_excepthook",
    "launch_ipdb_on_exception",
]

LAUNCH_BLOCKING_ENV = "CUDA_LAUNCH_BLOCKING"
SNAPSHOT_DIR_ENV = "IPDB_GPU_SNAPSHOT_DIR"

_recording = False
_snapshot_dir = None


def _canonical():
    """Return the importable ``ipdb.gpu`` module object.

    ``python -m ipdb.gpu`` executes this file as ``__main__``, so a target
    script that later does ``import ipdb.gpu`` gets a *second* module object
    with its own globals. State written by the launcher would then be
    invisible to the script — ``dump_snapshot`` would claim nothing had been
    recorded when it had. The launcher therefore routes state-changing calls
    through the canonical import.
    """
    if __name__ == "__main__":
        import ipdb.gpu as canonical

        return canonical
    return sys.modules[__name__]


def _log(message):
    print("[ipdb.gpu] %s" % message, file=sys.stderr)


def _torch():
    """Return the torch module, or None when it is not importable.

    Deliberately does not import torch as a side effect of importing this
    module: the launcher has to set the environment first.
    """
    return sys.modules.get("torch") or _import_torch()


def _import_torch():
    try:
        import torch
    except Exception:
        return None
    return torch


def _cuda_ready(torch):
    """True when torch can actually talk to a GPU."""
    if torch is None:
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def enable_launch_blocking(quiet=False):
    """Set ``CUDA_LAUNCH_BLOCKING=1``.

    Returns True when the setting will take effect. Returns False — with a
    warning — when CUDA is already initialised, because the runtime has
    already read the variable and the process would run asynchronously while
    appearing to be configured otherwise. That false sense of safety is worse
    than not setting it, so it is reported loudly rather than swallowed.
    """
    torch = sys.modules.get("torch")
    already_initialised = False
    if torch is not None:
        try:
            already_initialised = torch.cuda.is_initialized()
        except Exception:
            already_initialised = False

    os.environ[LAUNCH_BLOCKING_ENV] = "1"

    if already_initialised:
        if not quiet:
            _log(
                "CUDA is already initialised, so %s=1 will NOT take effect in "
                "this process. Launch via `python -m ipdb.gpu your_script.py` "
                "to set it before torch loads." % LAUNCH_BLOCKING_ENV
            )
        return False

    if not quiet:
        _log("%s=1" % LAUNCH_BLOCKING_ENV)
    return True


def start_recording_memory(max_entries=100000, quiet=False):
    """Begin recording CUDA allocation history.

    ``max_entries`` is bounded by default: the torch default is effectively
    unlimited, which on a long training run costs real memory to hold the
    history that is supposed to help you debug running out of it.
    """
    global _recording

    torch = _torch()
    if not _cuda_ready(torch):
        if not quiet:
            _log("no CUDA device available; not recording memory history")
        return False

    record = getattr(torch.cuda.memory, "_record_memory_history", None)
    if record is None:
        if not quiet:
            _log(
                "torch %s has no _record_memory_history; memory snapshots "
                "unavailable" % torch.__version__
            )
        return False

    try:
        record(max_entries=max_entries)
    except Exception as error:
        if not quiet:
            _log("could not start memory recording: %r" % (error,))
        return False

    _recording = True
    if not quiet:
        _log("recording CUDA memory history (max_entries=%d)" % max_entries)
    return True


def _default_snapshot_path():
    directory = _snapshot_dir or os.environ.get(SNAPSHOT_DIR_ENV) or os.getcwd()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(directory, "cuda-memory-%s-pid%d.pickle" % (stamp, os.getpid()))


def dump_snapshot(path=None, quiet=False):
    """Write a CUDA memory snapshot, returning its path or None.

    Never raises. This runs on the failure path, where the caller already has
    an exception worth more than the snapshot — letting a dump error propagate
    would replace a real traceback with a diagnostic one.
    """
    torch = _torch()
    if not _cuda_ready(torch):
        return None

    dump = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dump is None:
        return None

    if not _recording and not quiet:
        _log(
            "memory history was never recorded, so the snapshot will be empty; "
            "call ipdb.gpu.enable() before the allocation you care about"
        )

    path = path or _default_snapshot_path()
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        dump(path)
    except Exception as error:
        if not quiet:
            _log("could not dump memory snapshot: %r" % (error,))
        return None

    if not quiet:
        _log("memory snapshot written to %s" % path)
        _log("open it at https://pytorch.org/memory_viz")
    return path


def _dump_then(hook):
    """Wrap an excepthook so a snapshot is written before it runs."""

    def excepthook(exc_type, value, traceback):
        try:
            dump_snapshot()
        except Exception:
            # Diagnostics must not displace the failure being diagnosed.
            pass
        return hook(exc_type, value, traceback)

    return excepthook


def _interactive():
    """True when there is a terminal to drive a debugger with.

    Dropping into post-mortem under nohup, a CI runner, or a piped stdout
    hangs the process on a prompt nobody can answer, so the snapshot is
    written and the debugger skipped instead.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def install_excepthook(post_mortem=True):
    """Dump a snapshot on any uncaught exception, then optionally debug.

    Deliberately does not go through ``ipdb.__main__.wrap_sys_excepthook``:
    that installs IPython's ``BdbQuit_excepthook``, which has been deprecated
    since IPython 5.1 and now raises unconditionally when called.
    """
    import traceback as traceback_module

    previous = sys.excepthook

    def hook(exc_type, value, tb):
        traceback_module.print_exception(exc_type, value, tb)

        if not post_mortem:
            return
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            return
        if not _interactive():
            _log("not a tty; skipping post-mortem (snapshot was still written)")
            return

        from ipdb.__main__ import post_mortem as _post_mortem

        _post_mortem(tb)

    hook.__wrapped_excepthook__ = previous
    sys.excepthook = _dump_then(hook)


def enable(
    launch_blocking=True,
    record_memory=True,
    max_entries=100000,
    snapshot_dir=None,
    excepthook=True,
    post_mortem=True,
    sync_debug_mode=None,
    quiet=False,
):
    """Turn on the torch debugging aids in one call.

    ``sync_debug_mode`` is off by default: "warn" reports implicit
    host-device synchronisations, which is invaluable when hunting a stall
    and pure noise otherwise.
    """
    global _snapshot_dir

    if snapshot_dir:
        _snapshot_dir = snapshot_dir

    if launch_blocking:
        enable_launch_blocking(quiet=quiet)

    if record_memory:
        start_recording_memory(max_entries=max_entries, quiet=quiet)

    if sync_debug_mode is not None:
        torch = _torch()
        setter = getattr(torch.cuda, "set_sync_debug_mode", None) if torch else None
        if setter is None:
            if not quiet:
                _log("torch has no set_sync_debug_mode; skipping")
        else:
            try:
                setter(sync_debug_mode)
                if not quiet:
                    _log("sync debug mode = %r" % (sync_debug_mode,))
            except Exception as error:
                if not quiet:
                    _log("could not set sync debug mode: %r" % (error,))

    if excepthook:
        install_excepthook(post_mortem=post_mortem)


class launch_ipdb_on_exception(object):
    """Context manager: dump a snapshot, then drop into ipdb, on exception."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, value, traceback):
        if exc_type is None:
            return False
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            return False

        dump_snapshot()

        from ipdb.__main__ import post_mortem

        import traceback as traceback_module

        traceback_module.print_exception(exc_type, value, traceback)
        if _interactive():
            post_mortem(traceback)
        else:
            _log("not a tty; skipping post-mortem (snapshot was still written)")
        return True


def main(argv=None):
    """``python -m ipdb.gpu [-m module | script.py] [args...]``

    Sets CUDA_LAUNCH_BLOCKING before the target — and therefore before torch —
    is imported, which is the only way the variable reliably applies.

    The ``-m`` form exists so the target can itself be a runner that owns the
    failure, such as ``-m pytest``, ``-m unittest`` or ``-m pdbtest``. Those
    runners cannot set the variable themselves: by the time a test body runs,
    torch is loaded and CUDA is initialised, so it would be read too late.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(main.__doc__.strip(), file=sys.stderr)
        return 2

    os.environ[LAUNCH_BLOCKING_ENV] = "1"
    _log("%s=1 (set before the target imports torch)" % LAUNCH_BLOCKING_ENV)

    canonical = _canonical()
    canonical._install_deferred_recording()
    canonical.install_excepthook(post_mortem=True)

    if argv[0] == "-m":
        if len(argv) < 2:
            print("-m requires a module name", file=sys.stderr)
            return 2
        module = argv[1]
        sys.argv = argv[1:]
        # Match `python -m`, which puts the working directory on the path.
        sys.path.insert(0, os.getcwd())
        try:
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        except SystemExit:
            raise
        return 0

    script = argv[0]
    sys.argv = argv
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit:
        raise
    return 0


def _install_deferred_recording():
    """Start memory recording as soon as torch initialises CUDA.

    torch is not importable yet when the launcher starts, so this hooks the
    import system and patches in a callback rather than importing torch early
    (which would defeat the whole point of the launcher).
    """
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "torch":
                return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            original_exec = spec.loader.exec_module

            def exec_module(module):
                original_exec(module)
                try:
                    _canonical().start_recording_memory()
                except Exception:
                    pass

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())


if __name__ == "__main__":
    sys.exit(main())
