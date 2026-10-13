"""Bounded recovery of provider-confirmed ineffective compactions.

Use real usage for growth: preflight estimates and provider counts can differ by
more than the growth allowance without any new material. Keep a high-water mark
across unsuccessful probes so shrinking and regrowing the same prompt cannot
re-arm indefinitely. A provider rejection also earns one emergency probe between
healthy usage readings; it does not disable structural or summary-failure guards.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def recover_after_growth(compressor: Any, tokens: int) -> None:
    baseline = compressor._ineffective_growth_tokens
    growth = max(1024, min(4096, int(compressor.context_length * 0.05)))
    if compressor._tripped() and baseline > 0 and tokens - baseline >= growth:
        compressor._ineffective_growth_tokens = tokens
        _grant_probe(compressor, "provider-confirmed prompt growth")


def recover_after_overflow(compressor: Any, *, trigger: str, bypass_cooldown: bool, force: bool) -> None:
    """The outer overflow path gets one probe, including after a process rebuild.

    Only the built-in compressor owns these guards. Keep plugin engines unchanged,
    and do not spend the probe while an independent structural delay blocks it.
    The caller's existing attempt budget still bounds overflow recovery.
    """
    from agent.context_compressor import ContextCompressor

    if trigger != "overflow" or not bypass_cooldown or force or not isinstance(compressor, ContextCompressor):
        return
    compressor._refresh_durable_guards()
    if (compressor._tripped() and not compressor._ineffective_overflow_probe_used
            and compressor._structural_no_op_backoff_until <= time.monotonic()):
        compressor._ineffective_overflow_probe_used = True
        _grant_probe(compressor, "provider context overflow")


def _grant_probe(compressor: Any, reason: str) -> None:
    compressor._record_ineffective_compression_verdict(1)
    compressor._set_anti_thrash_recovery_deadline(0.0)
    logger.info("Allowing one ineffective-compression recovery probe after %s", reason)


def apply_real_prompt_verdict(compressor: Any) -> None:
    """Judge a compaction once, against provider usage; ordinary growth may earn a probe."""
    tokens = compressor.last_prompt_tokens
    if tokens <= 0:
        return
    compressor.last_real_prompt_tokens = tokens
    compressor._provider_omits_usage = False
    if tokens < compressor.threshold_tokens:
        compressor._record_ineffective_compression_verdict(0)
    elif compressor._verify_compaction_cleared_threshold:
        compressor._record_ineffective_compression_verdict(compressor._ineffective_compression_count + 1)
        compressor._ineffective_growth_tokens = max(compressor._ineffective_growth_tokens, tokens)
        if not compressor.quiet_mode:
            logger.warning(
                "Compaction did not clear the threshold: %d real tokens still >= %d. "
                "ineffective_compression_count=%d",
                tokens, compressor.threshold_tokens, compressor._ineffective_compression_count,
            )
    else:
        recover_after_growth(compressor, tokens)
