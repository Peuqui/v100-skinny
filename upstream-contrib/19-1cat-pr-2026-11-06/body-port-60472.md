Title: [Bugfix][Core] Backport vllm-project/vllm#60472: close MessageQueue zmq resources via finalizer to avoid exit hang

## Purpose

Backport of vllm-project/vllm#60472 ([Bugfix] Close MessageQueue zmq resources via finalizer to avoid exit, upstream commit 9bbc58156, by Aarushi Jain), in the same form as the earlier backports of #44082 and #44744. main does not have it.

Each MessageQueue owns a private ZMQ context whose sockets are never closed explicitly. When a queue is only reachable through a reference cycle, the cyclic garbage collector can collect the context together with its sockets. CPython clears the context's weak socket references first, so pyzmq's Context.__del__ -> destroy() closes nothing and term() then blocks forever on the still-open sockets. The upstream fix registers a weakref.finalize per MessageQueue that holds the context and the sockets, closes the sockets with linger=0 and terminates the context; atexit stays off, so queues still alive at shutdown behave as before.

How we hit it (on our fork; the code involved, WorkerProc.wait_for_ready and MessageQueue, is the same on main): a pipeline-parallel boot (PP5) whose last stage ran out of CUDA memory while loading the drafter. The first stage had already reported READY (its death-pipe monitor logged "Parent process exited"), so WorkerProc.wait_for_ready had connected a response MessageQueue for it; that queue stayed in the failure traceback. The engine core logged "WorkerProc initialization failed" with its traceback and then never exited: at interpreter exit ray.shutdown() (atexit) imports ray.dag, that import triggers a garbage collection, and the orphaned context blocked in term(). The API server kept waiting for the engine, and our process manager reported the model as starting until its health-check timeout. py-spy of the hung engine core (pyzmq 27.2.0, libzmq 4.3.5, ray 2.58.0, Python 3.12.3):

term (zmq/sugar/context.py:264)
destroy (zmq/sugar/context.py:322)   sockets: []
__del__ (zmq/sugar/context.py:140)
_compile_bytecode (<frozen importlib._bootstrap_external>:753)   name: "ray.dag.dag_node"
<module> (ray/dag/__init__.py:1)
shutdown (ray/_private/worker.py:2109)

The source change is the upstream diff unchanged; it applies to main as is. The test file on main is older than upstream's, so the upstream test test_reader_in_reference_cycle_is_collected_without_blocking is added by hand (with the gc and weakref imports), at the same place as upstream, before test_warning_logs.

Not a duplicate: I searched the open and closed PRs and issues of this repository for 60472, "MessageQueue finalizer", "zmq resources", "exit hang" and shm_broadcast; none backports this fix. Upstream, vllm-project/vllm#47422 (open) changes WorkerProc.wait_for_ready to forward the worker's traceback; it does not overlap with this change.

AI assistance was used for this backport. I reviewed every line and ran the tests and boots below.

## Test Plan

python -m pytest tests/distributed/test_shm_broadcast.py
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
End to end on our fork (same patch): facebook/opt-125m with dummy weights, PP2 on two V100s, the API server started directly; a sitecustomize hook makes Worker.load_model on rank 1 raise torch.OutOfMemoryError 20 s after it starts, so rank 0 is READY first. Then a normal boot of the same model with one completion request.

## Test Result

[FILL: PR tree results]

End to end (fork, same patch):
- Without the change: the failure is logged 48 s after start; 60 s later the engine core and the API server are still alive (engine core main thread in the stack above).
- With the change: the failure is logged after 49 s, the API server exits with code 1 after 61 s ("Engine core initialization failed"), no process of the session is left.
- Normal boot with the change: healthy after 35 s, the completion request is served, SIGTERM shuts it down without leftover processes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK
