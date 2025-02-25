import asyncio
import signal
import time
from collections import defaultdict
from types import FrameType
from typing import Any

from atproto import (
    CAR,
    AsyncFirehoseSubscribeReposClient,
    AtUri,
    firehose_models,
    models,
    parse_subscribe_repos_message,
)

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (run once)
nltk.download("vader_lexicon")

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    """Analyze the sentiment of the given text and return 'positive', 'negative', or 'neutral'."""
    try:
        scores = sid.polarity_scores(text)
        if scores["compound"] > 0.05:
            return "positive"
        elif scores["compound"] < -0.05:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return "unknown"


_INTERESTED_RECORDS = {
    models.ids.AppBskyFeedLike: models.AppBskyFeedLike,
    models.ids.AppBskyFeedPost: models.AppBskyFeedPost,
    models.ids.AppBskyGraphFollow: models.AppBskyGraphFollow,
}


def _get_ops_by_type(commit: models.ComAtprotoSyncSubscribeRepos.Commit) -> defaultdict:
    operation_by_type = defaultdict(lambda: {"created": [], "deleted": []})
    car = CAR.from_bytes(commit.blocks)
    for op in commit.ops:
        if op.action == "update":
            continue
        uri = AtUri.from_str(f"at://{commit.repo}/{op.path}")
        if op.action == "create":
            if not op.cid:
                continue
            create_info = {"uri": str(uri), "cid": str(op.cid), "author": commit.repo}
            record_raw_data = car.blocks.get(op.cid)
            if not record_raw_data:
                continue
            record = models.get_or_create(record_raw_data, strict=False)
            record_type = _INTERESTED_RECORDS.get(uri.collection)
            if record_type and models.is_record_type(record, record_type):
                operation_by_type[uri.collection]["created"].append(
                    {"record": record, **create_info}
                )
        if op.action == "delete":
            operation_by_type[uri.collection]["deleted"].append({"uri": str(uri)})
    return operation_by_type


def measure_events_per_second(func: callable) -> callable:
    def wrapper(*args) -> Any:
        wrapper.calls += 1
        cur_time = time.time()
        if cur_time - wrapper.start_time >= 1:
            print(f"NETWORK LOAD: {wrapper.calls} events/second")
            wrapper.start_time = cur_time
            wrapper.calls = 0
        return func(*args)

    wrapper.calls = 0
    wrapper.start_time = time.time()
    return wrapper


async def signal_handler(_: int, __: FrameType) -> None:
    print("Keyboard interrupt received. Stopping...")
    await client.stop()


async def main(firehose_client: AsyncFirehoseSubscribeReposClient) -> None:
    @measure_events_per_second
    async def on_message_handler(message: firehose_models.MessageFrame) -> None:
        commit = parse_subscribe_repos_message(message)
        if not isinstance(commit, models.ComAtprotoSyncSubscribeRepos.Commit):
            return
        if commit.seq % 20 == 0:
            firehose_client.update_params(
                models.ComAtprotoSyncSubscribeRepos.Params(cursor=commit.seq)
            )
        if not commit.blocks:
            return
        ops = _get_ops_by_type(commit)
        for created_post in ops[models.ids.AppBskyFeedPost]["created"]:
            author = created_post["author"]
            record = created_post["record"]
            sentiment = analyze_sentiment(record.text)
            inlined_text = record.text.replace("\n", " ")
            print(
                f"[CREATED_AT={record.created_at}][AUTHOR={author}][SENTIMENT={sentiment}]: {inlined_text}"
            )

    await client.start(on_message_handler)


if __name__ == "__main__":
    signal.signal(
        signal.SIGINT, lambda _, __: asyncio.create_task(signal_handler(_, __))
    )
    start_cursor = None
    params = None
    if start_cursor is not None:
        params = models.ComAtprotoSyncSubscribeRepos.Params(cursor=start_cursor)
    client = AsyncFirehoseSubscribeReposClient(params)
    asyncio.get_event_loop().run_until_complete(main(client))
