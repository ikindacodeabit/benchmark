# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the runaway guards and the scratchpad -- each on AND off.

Run from the repository root (the module uses absolute `evaluation.*` imports):
    python -m pytest tests/evaluation/test_rlm_guards.py
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

    def test_the_cap_is_honored_and_the_marker_appears_once(self):
        """The old loop prepended a marker on every shrink pass -- stacking one
        copy per iteration -- and refused to shrink below 200 chars, so a small
        cap was silently ignored."""
        rlm = RLM(root_client=_ScriptedClient([""]), scratchpad=Scratchpad(max_notes_tokens=10))
        block = rlm._notes_block(["padding " * 500])
        self.assertEqual(block.count("oldest notes truncated"), 1)
        self.assertLessEqual(rlm.tok.count(block), 10)


class _ContextOverflow(Exception):
    status_code = 400

    def __str__(self):
        return "This model's maximum context length is 40960 tokens"


class _OverflowingClient(_ScriptedClient):
    """Rejects any prompt over `limit` characters, as a served model would."""

    def __init__(self, replies, limit: int):
        super().__init__(replies)
        self.limit = limit
        self.rejections = 0

    def chat(self, messages, **kw):
        if sum(len(m["content"]) for m in messages) > self.limit:
            self.rejections += 1
            raise _ContextOverflow()
        return super().chat(messages, **kw)


class RootContextOverflowTest(unittest.TestCase):
    """Without a --max-context-tokens budget the root's view grows every turn and
    nothing bounds it. One overflow used to end the example as an exception --
    the mechanism that zeroed an entire RULER-32k RLM run."""

    def test_the_server_rejecting_the_view_evicts_and_retries(self):
        # Distinct cells: three identical ones would trip the repetition breaker
        # instead, which is a different guard with its own test.
        chatty = [_code(f"print({c!r} * 3000)") for c in "abc"]
        root = _OverflowingClient(chatty + _finish("survived"), limit=9000)
        rlm = RLM(root_client=root, max_steps=8, exec_timeout=None)

        result = rlm.run("ctx", "task")

        self.assertGreater(root.rejections, 0, "the fixture should have overflowed at least once")
        self.assertEqual(result.answer, "survived")
        self.assertGreater(result.metrics["overflow_evictions"], 0)

    def test_an_unfixable_overflow_still_raises(self):
        """When there are no turns left to drop, the system prompt alone exceeds
        the window; dropping more cannot help and pretending otherwise would loop."""
        root = _OverflowingClient(_finish("never reached"), limit=10)
        rlm = RLM(root_client=root, max_steps=3, exec_timeout=None)

        with self.assertRaises(_ContextOverflow):
            rlm.run("ctx", "task")

    def test_a_non_overflow_error_is_not_retried(self):
        class _Boom(Exception):
            status_code = 500

        class _Client(_ScriptedClient):
            def chat(self, messages, **kw):
                raise _Boom()

        with self.assertRaises(_Boom):
            RLM(root_client=_Client([]), max_steps=3, exec_timeout=None).run("ctx", "task")


class GroundingTest(unittest.TestCase):
    """The guard has to reject guesses without rejecting honest answers."""

    def test_a_literal_merely_embedded_in_a_task_word_is_rejected(self):
        """Plain containment licensed any literal appearing ANYWHERE in the task
        string, including inside a longer number or word: a question mentioning
        2012 whitelisted the invented answer 12."""
        root = _ScriptedClient([_code("FINAL('12')")] * 4)
        rlm = RLM(root_client=root, max_steps=6, exec_timeout=None)

        result = rlm.run("nothing relevant", "How many were sold in 2012?")
        self.assertEqual(result.end_reason, "ungrounded_final")

    def test_a_word_genuinely_present_in_the_task_is_accepted(self):
        root = _ScriptedClient([_code("FINAL('beta')")])
        rlm = RLM(root_client=root, max_steps=3, exec_timeout=None)

        result = rlm.run("ctx", "Which of alpha or beta is it?")
        self.assertEqual(result.answer, "beta")

    def test_an_answer_in_the_elided_middle_of_a_long_output_is_grounded(self):
        """The observation shown to the model is truncated head+tail for display;
        grounding must read what actually printed, or a real answer buried in the
        middle reads as a fabrication."""
        root = _ScriptedClient(
            [
                _code("print('x' * 4000 + ' SECRET42 ' + 'y' * 4000)"),
                _code("FINAL('SECRET42')"),
            ]
        )
        rlm = RLM(root_client=root, max_steps=4, obs_limit=500, exec_timeout=None)

        result = rlm.run("ctx", "find the secret")
        self.assertEqual(result.answer, "SECRET42")
        self.assertIn("...[truncated", result.transcript[0]["observation"])

    def test_a_guess_dressed_as_an_fstring_is_still_a_guess(self):
        root = _ScriptedClient([_code('FINAL(f"42")')] * 4)
        rlm = RLM(root_client=root, max_steps=6, exec_timeout=None)

        self.assertEqual(rlm.run("ctx", "how many?").end_reason, "ungrounded_final")


class AbstentionTest(unittest.TestCase):
    """A model that searched and found nothing has something true to report, and
    no way to say it: FINAL("unknown") is by construction absent from the output
    it describes, so the grounding guard read the honest ending as a guess."""

    def test_final_none_ends_the_run_without_a_grounding_check(self):
        root = _ScriptedClient([_code("FINAL_NONE('the document never mentions it')")])
        rlm = RLM(root_client=root, max_steps=3, exec_timeout=None)

        result = rlm.run("ctx", "task")
        self.assertEqual(result.end_reason, "abstained")
        self.assertIsNone(result.answer)
        self.assertTrue(result.finished, "abstaining is a completed run, not a failure")
        self.assertEqual(result.metrics["abstain_reason"], "the document never mentions it")

    def test_final_none_is_honored_in_prose_too(self):
        root = _ScriptedClient(["I searched thoroughly. FINAL_NONE('not present')"])
        rlm = RLM(root_client=root, max_steps=3, exec_timeout=None)

        result = rlm.run("ctx", "task")
        self.assertEqual(result.end_reason, "abstained")
        self.assertIsNone(result.answer)


class ReplyParsingTest(unittest.TestCase):
    def test_a_code_block_cut_off_at_max_tokens_still_runs(self):
        """A reply truncated mid-block has an opening fence and no closing one.
        Read as 'no code block' it burned a nudge, and three of them returned the
        raw truncated text as the answer."""
        root = _ScriptedClient(["Here goes:\n```python\nans = 'found it'\nprint(ans)"] + [_code("FINAL(ans)")])
        rlm = RLM(root_client=root, max_steps=4, exec_timeout=None)

        result = rlm.run("ctx", "task")
        self.assertEqual(result.answer, "found it")
        self.assertEqual(result.metrics["truncated_code_blocks"], 1)

    def test_final_var_on_a_missing_variable_is_not_an_answer(self):
        """It used to answer the literal string '<missing var x>' and record it as
        a finished, successful prediction."""
        root = _ScriptedClient([_code("FINAL_VAR('nope')")] + _finish("real answer"))
        rlm = RLM(root_client=root, max_steps=5, exec_timeout=None)

        result = rlm.run("ctx", "task")
        self.assertIn("NameError", result.transcript[0]["observation"])
        self.assertEqual(result.answer, "real answer")

    def test_prose_final_var_naming_an_unset_variable_falls_through_to_a_nudge(self):
        root = _ScriptedClient(["I am done: FINAL_VAR('never_set')"] + _finish("real answer"))
        rlm = RLM(root_client=root, max_steps=5, exec_timeout=None)

        result = rlm.run("ctx", "task")
        self.assertNotIn("missing var", str(result.answer))
        self.assertEqual(result.answer, "real answer")


if __name__ == "__main__":
    unittest.main()
