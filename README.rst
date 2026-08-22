IPython `pdb`
=============

.. image:: https://github.com/gotcha/ipdb/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/gotcha/ipdb/actions/workflows/tests.yml
.. image:: https://codecov.io/gh/gotcha/ipdb/branch/master/graphs/badge.svg?style=flat
  :target: https://codecov.io/gh/gotcha/ipdb?branch=master

Use
---

ipdb exports functions to access the IPython_ debugger, which features
tab completion, syntax highlighting, better tracebacks, better introspection
with the same interface as the `pdb` module.

Example usage:

.. code-block:: python

        import ipdb
        ipdb.set_trace()
        ipdb.set_trace(context=5)  # will show five lines of code
                                   # instead of the default three lines
                                   # or you can set it via IPDB_CONTEXT_SIZE env variable
                                   # or setup.cfg file
        ipdb.pm()
        ipdb.run('x[0] = 3')
        result = ipdb.runcall(function, arg0, arg1, kwarg='foo')
        result = ipdb.runeval('f(1,2) - 3')


Arguments for `set_trace`
+++++++++++++++++++++++++

The `set_trace` function accepts `context` which will show as many lines of code as defined,
and `cond`, which accepts boolean values (such as `abc == 17`) and will start ipdb's
interface whenever `cond` equals to `True`.

Using configuration file
++++++++++++++++++++++++

It's possible to set up context using a `.ipdb` file on your home folder, `setup.cfg`
or `pyproject.toml` on your project folder. You can also set your file location via
env var `$IPDB_CONFIG`. Your environment variable has priority over the home
configuration file, which in turn has priority over the setup config file.
Currently, only context setting is available.

A valid setup.cfg is as follows

::

        [ipdb]
        context=5


A valid .ipdb is as follows

::

        context=5


A valid pyproject.toml is as follows

::

        [tool.ipdb]
        context=5


The post-mortem function, ``ipdb.pm()``, is equivalent to the magic function
``%debug``.

.. _IPython: http://ipython.org

If you install ``ipdb`` with a tool which supports ``setuptools`` entry points,
an ``ipdb`` script is made for you. You can use it to debug your python 2 scripts like

::

        $ bin/ipdb mymodule.py

And for python 3

::

        $ bin/ipdb3 mymodule.py

Alternatively with Python 2.7 only, you can also use

::

        $ python -m ipdb mymodule.py

You can also enclose code with the ``with`` statement to launch ipdb if an exception is raised:

.. code-block:: python

        from ipdb import launch_ipdb_on_exception

        with launch_ipdb_on_exception():
            [...]

.. warning::
   Context managers were introduced in Python 2.5.
   Adding a context manager implies dropping Python 2.4 support.
   Use ``ipdb==0.6`` with 2.4.

Or you can use ``iex`` as a function decorator to launch ipdb if an exception is raised:

.. code-block:: python

        from ipdb import iex

        @iex
        def main():
            [...]

.. warning::
   Using ``from future import print_function`` for Python 3 compat implies dropping Python 2.5 support.
   Use ``ipdb<=0.8`` with 2.5.

Issues with ``stdout``
----------------------

Some tools, like ``nose`` fiddle with ``stdout``.

Until ``ipdb==0.9.4``, we tried to guess when we should also
fiddle with ``stdout`` to support those tools.
However, all strategies tried until 0.9.4 have proven brittle.

If you use ``nose`` or another tool that fiddles with ``stdout``, you should
explicitly ask for ``stdout`` fiddling by using ``ipdb`` like this

.. code-block:: python

        import ipdb
        ipdb.sset_trace()
        ipdb.spm()

        from ipdb import slaunch_ipdb_on_exception
        with slaunch_ipdb_on_exception():
            [...]


PyTorch and GPU debugging
-------------------------

A Python debugger only sees the host side of a PyTorch program. CUDA kernel
launches are asynchronous, so a device-side error is raised at whatever line
the host happened to reach — not the line that caused it — and a breakpoint
says nothing about what the GPU is doing.

``ipdb.gpu`` addresses the two parts of that which have to be arranged
*before* the failure.

Run a script through the launcher

.. code-block:: console

        python -m ipdb.gpu your_script.py [args...]

which sets ``CUDA_LAUNCH_BLOCKING=1`` before anything imports torch, starts
recording CUDA allocation history as soon as torch loads, and dumps a memory
snapshot on any uncaught exception before dropping into ``ipdb``.

The timing matters. The CUDA runtime reads ``CUDA_LAUNCH_BLOCKING`` when it
creates a context, so setting it after CUDA has initialised does nothing at
all. In a real out-of-bounds indexing failure the difference is the traceback
pointing at ``src[idx]`` rather than at the ``torch.cuda.synchronize()`` three
lines later.

Or enable it in-process, before the first CUDA call

.. code-block:: python

        import ipdb.gpu
        ipdb.gpu.enable()

``enable()`` reports whether ``CUDA_LAUNCH_BLOCKING`` actually took effect
rather than assuming it did — if CUDA is already initialised it says so
loudly, because believing launches are synchronous when they are not is worse
than knowing they are asynchronous.

Snapshots are written next to the process (or to ``snapshot_dir``, or the
directory in ``IPDB_GPU_SNAPSHOT_DIR``) and open in the PyTorch memory
viewer at https://pytorch.org/memory_viz, where each allocation carries the
Python stack that made it.

Other useful arguments to ``enable()``:

``sync_debug_mode="warn"``
    Report implicit host-device synchronisations — invaluable when chasing a
    stall, noise otherwise, so it is off by default.

``post_mortem=False``
    Write the snapshot but do not open a debugger. Post-mortem is skipped
    automatically when stdin is not a terminal, so unattended runs do not hang
    on a prompt nobody can answer.

``max_entries``
    Bound on the recorded allocation history. Defaults to 100k rather than
    torch's effectively-unlimited default, which on a long run spends real
    memory to hold the history meant to diagnose running out of it.

Everything degrades quietly: ``ipdb.gpu`` imports without torch installed,
no-ops without a CUDA device, and a snapshot that cannot be written is
reported rather than raised — it runs on the failure path, where the
exception already in flight is worth more than the diagnostics.

Development
-----------

``ipdb`` source code and tracker are at https://github.com/gotcha/ipdb.

Pull requests should take care of updating the changelog ``HISTORY.txt``.

Under the unreleased section, add your changes and your username.

Manual testing
++++++++++++++

To test your changes, make use of ``manual_test.py``. Create a virtual environment,
install IPython and run ``python manual_test.py`` and check if your changes are in effect.
If possible, create automated tests for better behaviour control.

Automated testing
+++++++++++++++++

To run automated tests locally, create a virtual environment, install `coverage`
and run `coverage run setup.py test`.

Third-party support
-------------------

pytest
+++++++
pytest_ supports a ``--pdb`` option which can run ``ipdb`` /
``IPython.terminal.debugger:TerminalPdb`` on ``Exception`` and ``breakpoint()``:

.. code:: bash

    pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb -v ./test_example.py

You don't need to specify ``--pdbcls`` for every ``pytest`` invocation 
if you add ``addopts`` to ``pytest.ini`` or ``pyproject.toml``.

``pytest.ini``:

.. code:: bash

  [tool.pytest.ini_options]
  addopts = "--pdbcls=IPython.terminal.debugger:TerminalPdb"

``pyproject.toml``:

.. code:: yml

  [tool.pytest.ini_options]
  addopts = "--pdbcls=IPython.terminal.debugger:TerminalPdb"


.. _pytest: https://pypi.python.org/pypi/pytest
