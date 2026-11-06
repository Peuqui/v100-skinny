## Purpose

When one worker fails during WorkerProc initialization after other workers have already reported READY, the engine core process never exits. The API server then waits for the engine forever instead of failing fast; under a process manager with a health-check timeout the model sits in "starting" until that timeout. We hit this with a pipeline-parallel boot (PP5) whose last stage ran out of CUDA memory while loading the drafter: the other four stages were READY, the engine core logged "WorkerProc initialization failed" and its traceback, and then stayed alive, sleeping.

Cause: WorkerProc.wait_for_ready connects a response MessageQueue (its own ZMQ context with a SUB socket) for every worker as soon as that worker reports READY. When a later worker fails, wait_for_ready raises, and those queues stay referenced from the exception's traceback, i.e. from a reference cycle. At interpreter exit, ray.shutdown() (registered with atexit by ray) imports ray.dag, which triggers a garbage collection. The collector finalizes one of the orphaned ZMQ contexts; Context.__del__ calls destroy(), which no longer sees the socket (the weak references are already cleared) and then blocks in zmq_ctx_term forever, because the socket is still open. py-spy of the hung engine core (pyzmq 27.2.0, libzmq 4.3.5, ray 2.58.0, Python 3.12.3):

term (zmq/sugar/context.py:264)
destroy (zmq/sugar/context.py:322)   sockets: []
__del__ (zmq/sugar/context.py:140)
_compile_bytecode (<frozen importlib._bootstrap_external>:753)   name: "ray.dag.dag_node"
...
<module> (ray/dag/__init__.py:1)
...
shutdown (ray/_private/worker.py:2109)

A failure later in startup (for example in the profile run) is not affected: there the executor already holds its queues and shuts them down.

Change: wait_for_ready first collects the READY responses of all workers and connects the response queues only once every worker is ready. A failed start then holds no ZMQ sockets. The ready responses are kept in a dict by worker index, so the cast of the old list of optional handles is gone. Workers do not depend on the earlier connect: after sending READY each worker first waits for the executor's broadcast queue, which the executor opens only after wait_for_ready returns.

Not a duplicate: I searched the open and closed PRs of this repository for wait_for_ready, "hang worker start" and zmq, and the vLLM PRs for wait_for_ready. vllm-project/vllm#47422 (open) changes the same function, but for a different problem: it forwards the worker's traceback over the ready pipe so the parent can show the root cause. It does not touch when the response queues are connected, so the hang remains with it. The two changes edit different lines of wait_for_ready.

AI assistance was used for this change. I reviewed every line and ran the tests and boots below.

## Test Plan

python -m pytest tests/v1/executor/test_multiproc_wait_for_ready.py
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
New tests: test_wait_for_ready_connects_no_queue_when_a_later_worker_fails (worker 0 reports READY, worker 1 dies after that READY has been read; no response queue may be connected) and test_wait_for_ready_connects_every_worker_in_rank_order (both READY, queues connected for both, handles in rank order). The first test fails on main and passes with this change.
End to end on our fork: facebook/opt-125m with dummy weights, PP2 on two V100s, the API server started directly; a sitecustomize hook makes Worker.load_model on rank 1 raise torch.OutOfMemoryError 20 s after it starts, so rank 0 is READY first. Then a normal boot of the same model with one completion request.

## Test Result

[FILL: PR tree results]

End to end (fork, same patch):
- Without the change: the failure is logged 48 s after start; 60 s later the engine core and the API server are still alive (engine core state Sl, main thread in the stack above).
- With the change: the failure is logged after 47 s, the API server exits with code 1 after 58 s ("Engine core initialization failed"), no process of the session is left.
- Normal boot with the change: healthy after 33 s, the completion request is served, SIGTERM shuts it down without leftover processes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK
