# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the runaway guards and the scratchpad -- each on AND off.

Run from the repository root (the module uses absolute `evaluation.*` imports):
    python -m pytest evaluation/rlm/test_guards.py
"""

import time
import unittest

from evaluation.rlm.rlm import RLM, MemoryBudget, Scratchpad


class _ScriptedClient:
    """Returns canned replies in order, then loops on the last one."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list = []
        self.calls = 0

    def chat(self, messages, **kw):
        self.seen.append(messages)
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


def _code(body: str) -> str:
    return f"```python\n{body}\n```"


# The harness refuses FINAL() on a literal the model never saw in real REPL output
# (its anti-guessing check), so a fixture that wants to finish must compute and
# print the answer first, then FINAL the variable holding it.
def _finish(value: str) -> list[str]:
    return [_code(f"ans = {value!r}\nprint(ans)"), _code("FINAL(ans)")]


class ExecTimeoutTest(unittest.TestCase):
    def test_a_runaway_loop_is_aborted_and_reported_to_the_model(self):
        root = _ScriptedClient([_code("while True:\n    pass")] + _finish("done"))
        rlm = RLM(root_client=root, max_steps=4, exec_timeout=1.0)

        started = time.monotonic()
        result = rlm.run("some context", "do the thing")
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 20, "the infinite loop should have been cut short")
        self.assertIn("[TIMEOUT]", result.transcript[0]["observation"])
        # The model gets to see the timeout and recover, rather than the run dying.
        self.assertEqual(result.answer, "done")

    def test_exec_timeout_none_disables_the_watchdog(self):
        root = _ScriptedClient(_finish("ok"))
        rlm = RLM(root_client=root, max_steps=3, exec_timeout=None)

        result = rlm.run("ctx", "task")

        self.assertNotIn("[TIMEOUT]", result.transcript[0]["observation"])
        self.assertEqual(result.answer, "ok")

    def test_model_code_calling_exit_does_not_kill_the_sweep(self):
        root = _ScriptedClient([_code("import sys\nsys.exit(1)")] + _finish("survived"))
        rlm = RLM(root_client=root, max_steps=4)

        result = rlm.run("ctx", "task")

        self.assertIn("[EXCEPTION]", result.transcript[0]["observation"])
        self.assertEqual(result.answer, "survived")


class RunTimeoutTest(unittest.TestCase):
    def test_deadline_ends_the_example_with_its_own_end_reason(self):
        class _SlowClient(_ScriptedClient):
            def chat(self, messages, **kw):
                time.sleep(0.25)
                return super().chat(messages, **kw)

        root = _SlowClient([_code("print('still going')")])
        rlm = RLM(root_client=root, max_steps=50, run_timeout=0.5)

        result = rlm.run("ctx", "task")

        self.assertEqual(result.end_reason, "run_timeout")
        self.assertFalse(result.finished)
        self.assertIsNone(result.answer)
        self.assertLess(root.calls, 50, "should have stopped well before max_steps")

    def test_run_timeout_none_disables_the_deadline(self):
        root = _ScriptedClient(_finish("quick"))
        rlm = RLM(root_client=root, max_steps=3, run_timeout=None)

        result = rlm.run("ctx", "task")

        self.assertEqual(result.end_reason, "final_called")
        self.assertEqual(result.answer, "quick")


class SubCallCapTest(unittest.TestCase):
    def test_cap_returns_a_notice_instead_of_hitting_the_api(self):
        root = _ScriptedClient(
            [
                _code("for i in range(5):\n    print(llm_query('q%d' % i))"),
                _code("FINAL('end')"),
            ]
        )
        sub = _ScriptedClient(["sub answer"])
        rlm = RLM(root_client=root, sub_client=sub, max_steps=3, max_sub_calls=2, cache_subcalls=False)

        result = rlm.run("ctx", "task")

        self.assertEqual(sub.calls, 2, "must stop calling the API at the cap")
        self.assertIn("[SUB-CALL LIMIT REACHED]", result.transcript[0]["observation"])

    def test_max_sub_calls_none_disables_the_cap(self):
        root = _ScriptedClient(
            [
                _code("for i in range(5):\n    print(llm_query('q%d' % i))"),
                _code("FINAL('end')"),
            ]
        )
        sub = _ScriptedClient(["sub answer"])
        rlm = RLM(root_client=root, sub_client=sub, max_steps=3, max_sub_calls=None, cache_subcalls=False)

        rlm.run("ctx", "task")

        self.assertEqual(sub.calls, 5)


class SubCallCacheKeyTest(unittest.TestCase):
    def test_prompts_sharing_a_truncated_prefix_do_not_collide(self):
        """Keying the cache on the TRUNCATED prompt returns one slice's answer for another."""
        long_a = "A" * 100 + "first tail"
        long_b = "A" * 100 + "second tail"
        root = _ScriptedClient(
            [
                _code(f"print(llm_query({long_a!r}))\nprint(llm_query({long_b!r}))"),
                _code("FINAL('end')"),
            ]
        )
        sub = _ScriptedClient(["answer one", "answer two"])
        # Truncation point falls inside the shared prefix, so both prompts are sent
        # to the sub-model as the identical string.
        rlm = RLM(root_client=root, sub_client=sub, max_steps=3, max_subcall_chars=50)

        result = rlm.run("ctx", "task")

        self.assertEqual(sub.calls, 2, "distinct prompts must not share a cache entry")
        observation = result.transcript[0]["observation"]
        self.assertIn("answer one", observation)
        self.assertIn("answer two", observation)


class ScratchpadTest(unittest.TestCase):
    def test_note_tool_absent_unless_enabled(self):
        root = _ScriptedClient([_code("note('x')"), _code("FINAL('end')")])
        rlm = RLM(root_client=root, max_steps=3)

        result = rlm.run("ctx", "task")

        self.assertIn("NameError", result.transcript[0]["observation"])
        self.assertNotIn("note(text", root.seen[0][0]["content"])

    def test_notes_persist_and_are_reinjected_every_turn(self):
        root = _ScriptedClient(
            [
                _code("note('the passkey is 8391')"),
                _code("print('thinking')"),
                _code("FINAL('8391')"),
            ]
        )
        rlm = RLM(root_client=root, max_steps=4, scratchpad=Scratchpad())

        result = rlm.run("ctx", "task")

        self.assertIn("note(text", root.seen[0][0]["content"], "the tool must be documented in the prompt")
        # By the third turn the note is still in the sent messages.
        third_turn = "".join(m["content"] for m in root.seen[2])
        self.assertIn("the passkey is 8391", third_turn)
        self.assertEqual(result.metrics["notes_saved"], 1)
        self.assertTrue(result.metrics["scratchpad"])

    def test_notes_survive_budget_eviction(self):
        root = _ScriptedClient(
            [
                _code("note('keep me')"),
                _code("print('x' * 4000)"),
                _code("print('y' * 4000)"),
                _code("FINAL('done')"),
            ]
        )
        rlm = RLM(
            root_client=root,
            max_steps=5,
            budget=MemoryBudget(max_context_tokens=1200, keep_recent_turns=1),
            scratchpad=Scratchpad(),
        )

        result = rlm.run("ctx", "task")

        self.assertGreater(result.metrics["evictions"], 0, "the budget should have evicted turns")
        last_turn = "".join(m["content"] for m in root.seen[-1])
        self.assertIn("keep me", last_turn, "the note must outlive the evicted turn that created it")

    def test_notes_block_is_capped(self):
        rlm = RLM(root_client=_ScriptedClient([""]), scratchpad=Scratchpad(max_notes_tokens=20))
        block = rlm._notes_block(["padding " * 200])
        self.assertIn("oldest notes truncated", block)
        self.assertLess(rlm.tok.count(block), 200)


if __name__ == "__main__":
    unittest.main()
