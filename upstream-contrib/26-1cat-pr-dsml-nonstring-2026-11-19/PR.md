GESENDET 19.11.2026 17:34 als 1CatAI/1Cat-vLLM#1571 (Kopf 2afa40e2d, Basis main 1fa9c438b; Live-Text geprüft = Entwurf)

Titel: [Bugfix][Parser] DeepSeek DSML: stream non-string tool arguments only when complete

Zweig: Peuqui:pr-dsml-nonstring-stream (Basis 1Cat main 1fa9c438b), Commits b99bb8e49 + 2afa40e2d
Fork-Fassung: a849e3760 / 540d9eaf8 (prod-v13), Entwicklung 4db9ef961 / 6c2c1f616

----- Text -----

Problem: The DeepSeek V3.2/V4 DSML parser streams an object or array parameter as raw text, but at finish extract_tool_calls coerces the same value through the tool schema (coerce_to_schema_type). When the model's text for such a value is not valid JSON, the batch path keeps it as a JSON string. The streamed arguments are then not a prefix of the final ones, and flush_tool_calls aborts the response with "Tool arguments at finish do not extend the streamed prefix". Examples that trigger it: ['a'] (single quotes), {"a": True}, [1, 2,], [1e999], or plain text for an array parameter.

A second mismatch of the same kind: an "arguments"/"input" wrapper parameter that is not in the schema is unwrapped by the streaming path as soon as it closes, while _repair_param_dict in the batch path unwraps it only when it is the sole parameter. A wrapper followed by another parameter therefore also ends in the same error.

Seen in production on DeepSeek-V4-Flash (deepseek_v4 tool parser, streaming with tools): a turn of an agent with file tools aborted with that ValueError from tool_stream.py remaining_tool_args via serving.py chat_completion_stream_generator -> flush_tool_calls. Reproduced on the CPU with the parser alone and the production finish check; Unicode in arguments is not involved (strings are written with ensure_ascii=False on both paths, and an escaped \u00fc inside raw JSON passes the value comparison).

Change (vllm/tool_parsers/deepseekv32_tool_parser.py):
- Non-string parameter values are buffered and sent once the parameter is complete, converted exactly as in the batch path. String values (string="true") still stream incrementally. This removes the raw streaming mode (_can_stream_raw_param, _param_types_for_name).
- A wrapper parameter is held until the next parameter opens (then sent as is) or the invoke ends (then unwrapped as in the batch path).
- _convert_params_with_schema is split into _coerce_params (per-parameter conversion) and the wrapper repair, so the streaming path converts single parameters without unwrapping them. DeepSeekV41ToolParser keeps its own conversion without unwrapping (deepseekv41_tool_parser.py).
- vllm/tool_parsers/utils.py: find_tool_properties takes a Sequence (the parsers pass their invariant tool list, mypy flagged it), and a variable reused with two types in get_json_schema_from_tools gets its own name.

This matches the design of the newer upstream DeepSeek parser (vllm/parser/deepseek_v4.py), which also sends non-string values only when complete and unwraps the wrapper on the whole arguments.

Second commit (tests only): the adjust_request tests of functiongemma and kimi_k2 fail on main with "Mock object has no attribute 'parallel_tool_calls'": adjust_request reads request.parallel_tool_calls since #1316, and a MagicMock(spec=ChatCompletionRequest) has no pydantic fields. The mocks now set it to its default (True). Included because tests/tool_parsers is not green without it.

Not a duplicate: of the 72 open PRs, only #1031 touches vllm/tool_parsers (it adds deepseekv41_engine_tool_parser.py); none changes deepseekv32_tool_parser.py, deepseekv41_tool_parser.py, utils.py, tool_stream.py or these tests. Searches for DSML, deepseek tool parser, streamed prefix and parallel_tool_calls find nothing else.

Tests, on main 1fa9c438b (CPU, CUDA_VISIBLE_DEVICES empty; pre-commit passes on the changed files, including ruff-check, ruff-format and mypy):
- New test_streamed_arguments_match_finish_time_arguments in tests/tool_parsers/test_deepseekv4_tool_parser.py: 11 cases (the problem cases above plus controls) streamed at chunk sizes 1, 3 and 7, each checked with remaining_tool_args against the batch result. Without the change 24 of 33 fail (all 8 problem cases at every chunk size), with it all 33 pass.
- tests/tool_parsers/test_deepseekv4_tool_parser.py, test_deepseekv32_tool_parser.py, test_deepseekv41_tool_parser.py and tests/entrypoints/openai/test_tool_stream_finish.py: 145 passed.
- tests/tool_parsers/test_functiongemma_tool_parser.py and test_kimi_k2_tool_parser.py: 4 failed before the second commit, 65 passed with it.
- All of tests/tool_parsers plus test_tool_stream_finish.py, one pytest process per file, on our fork with the same change: 44 of 48 files pass before the mock fix; the two mock files pass after it; test_llama3_json_tool_parser.py needs the gated meta-llama/Llama-3.2-1B-Instruct (403 here), and test_qwen3xml_tool_parser.py::test_malformed_input[True] fails the same way without this change.
- Live, on our fork with the same change (DeepSeek-V4-Flash, PP5 over 3x V100 + 2x Quadro RTX 8000, deepseek_v4 tool and reasoning parsers, streaming through the OpenAI API): 12 of 12 streamed tool-call turns well-formed (integer, boolean, object and array parameters, non-ASCII paths, two tools in one turn); every call's concatenated argument deltas parse as JSON with the schema's types, and the server log shows no "do not extend the streamed prefix" since. In these turns the model wrote valid JSON, so this shows no regression; the failing case itself is covered by the unit test above.

AI assistance (Claude) was used; I reviewed the change and ran the tests above.
