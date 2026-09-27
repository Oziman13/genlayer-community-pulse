# v0.2.16
# { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }

from genlayer import *

# CommunityPulse: a sentiment-gated moderation primitive
#
# Purpose
# -------
# Any community surface (a forum thread, a DAO proposal, a support inbox)
# can call submit_feedback(text) to record free-text feedback. The contract
# keeps a running, on-chain tally of how many submissions were positive,
# negative, or neutral, and exposes a configurable "flagged" state that
# other contracts, bots, or dashboards can react to (e.g. pause
# auto-posting, alert moderators, gate a feature rollout) once negative
# sentiment crosses a threshold. This is meant to be reused as a building
# block, not a one-off demo.
#
# Consensus / Equivalence Principle
# ----------------------------------
# Two LLM calls on the same text rarely match byte-for-byte
# ("Positive." vs "positive" vs "This reads as positive."), so requiring
# identical output (strict_eq) would reject perfectly good answers. Instead
# this contract uses gl.eq_principle.prompt_comparative: each validator
# independently classifies the same text, and an LLM judge decides whether
# the validator's answer and the leader's answer *mean the same thing* --
# rather than requiring identical bytes. This is the equivalence check
# GenLayer recommends for natural-language, non-deterministic output.
#
# Everything downstream of that single agreed-upon classification --
# bucketing it into positive/negative/neutral, updating counters, and
# recomputing the flag -- is plain, deterministic Python that needs no
# further validator judgment.

class CommunityPulse(gl.Contract):
    total_count: bigint
    positive_count: bigint
    negative_count: bigint
    neutral_count: bigint
    last_text: str
    last_sentiment: str
    flag_threshold_percent: bigint
    min_samples_before_flagging: bigint
    is_flagged_flag: bigint  # 0 or 1; bool avoided as a storage field on purpose

    # constructor: each community configures its own sensitivity
    def __init__(self, flag_threshold_percent: int, min_samples_before_flagging: int):
        self.total_count = bigint(0)
        self.positive_count = bigint(0)
        self.negative_count = bigint(0)
        self.neutral_count = bigint(0)
        self.last_text = ""
        self.last_sentiment = ""
        self.flag_threshold_percent = bigint(flag_threshold_percent)
        self.min_samples_before_flagging = bigint(min_samples_before_flagging)
        self.is_flagged_flag = bigint(0)

    # write method: classifies feedback via LLM consensus, then updates
    # deterministic on-chain aggregates and the moderation flag
    @gl.public.write
    def submit_feedback(self, text: str) -> None:
        prompt = (
            "Classify the sentiment of the text below.\n"
            "Respond with exactly one word, lowercase, no punctuation, "
            "no explanation: positive, negative, or neutral.\n\n"
            f"Text: {text}"
        )

        def classify() -> str:
            result = gl.nondet.exec_prompt(prompt)
            return result.strip().lower()

        # Comparative equivalence: an LLM judge checks that the leader's
        # and each validator's independent classification mean the same
        # thing, instead of demanding identical bytes.
        raw_sentiment = gl.eq_principle.prompt_comparative(
            classify,
            "The sentiment classification (positive, negative, or neutral) must match.",
        )

        sentiment = self._normalize(raw_sentiment)

        self.last_text = text
        self.last_sentiment = sentiment
        self.total_count = self.total_count + bigint(1)

        if sentiment == "positive":
            self.positive_count = self.positive_count + bigint(1)
        elif sentiment == "negative":
            self.negative_count = self.negative_count + bigint(1)
        else:
            self.neutral_count = self.neutral_count + bigint(1)

        self._refresh_flag()

    # deterministic helper: buckets any near-equivalent phrasing the
    # comparative principle accepted into exactly one of three categories
    def _normalize(self, raw: str) -> str:
        text = raw.strip().lower()
        if "pos" in text:
            return "positive"
        if "neg" in text:
            return "negative"
        return "neutral"

    # deterministic helper: recomputes the moderation flag from the
    # on-chain counters -- no LLM call involved
    def _refresh_flag(self) -> None:
        if self.total_count < self.min_samples_before_flagging:
            self.is_flagged_flag = bigint(0)
            return
        negative_ratio_x100 = (self.negative_count * bigint(100)) // self.total_count
        if negative_ratio_x100 >= self.flag_threshold_percent:
            self.is_flagged_flag = bigint(1)
        else:
            self.is_flagged_flag = bigint(0)

    # read methods must be annotated with view
    @gl.public.view
    def get_last_sentiment(self) -> str:
        return self.last_sentiment

    @gl.public.view
    def get_last_text(self) -> str:
        return self.last_text

    @gl.public.view
    def get_total_count(self) -> bigint:
        return self.total_count

    @gl.public.view
    def get_positive_count(self) -> bigint:
        return self.positive_count

    @gl.public.view
    def get_negative_count(self) -> bigint:
        return self.negative_count

    @gl.public.view
    def get_neutral_count(self) -> bigint:
        return self.neutral_count

    @gl.public.view
    def is_flagged_now(self) -> bool:
        return self.is_flagged_flag == bigint(1)
