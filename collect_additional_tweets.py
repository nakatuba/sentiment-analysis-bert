import argparse
import datetime
import hashlib
import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from retry import retry
from tqdm import tqdm

# Requests per 15-minute window unless otherwise stated
RATE_LIMITS_SLEEP = 60 * 15


@dataclass
class SearchTweetData(object):
    created_at: str
    text: str
    tweet_id: str
    author_id: str


@dataclass
class SearchTweetInclude(object):
    user_id: str
    name: str
    username: str


@dataclass
class SearchTweetIncludes(object):
    users: List[SearchTweetInclude]


@dataclass
class SearchTweetMeta(object):
    result_count: int
    newest_id: Optional[str]
    oldest_id: Optional[str]


@dataclass
class SearchTweetResult(object):
    data: List[SearchTweetData]
    includes: SearchTweetIncludes
    meta: SearchTweetMeta


def get_bearer_token() -> str:
    load_dotenv()
    return os.environ["BEARER_TOKEN"]


def bearer_oauth(
    r: requests.models.PreparedRequest, bearer_token: str
) -> requests.models.PreparedRequest:
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FullArchiveSearchPython"
    return r


def clean_text(s: str) -> str:
    s = s.replace("\\n", " ")
    return s


def get_tweet_datetime(
    s: str,
    datetime_formats: Tuple[str, ...] = (
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d/ %H:%M",
        "%Y/%m/%d %H/%M",
        "%Y/%m/%d, %H:%M",
        " %Y/%m/%d %H:%M",
        "  %Y/%m/%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d?%H:%M",
        "%Y/%m/%d t%H:%M",
    ),
) -> Optional[str]:

    dt: Optional[datetime.datetime] = None
    for f in datetime_formats:
        try:
            dt = datetime.datetime.strptime(s, f)
        except ValueError:
            continue

    if dt is None:
        breakpoint()
        return None

    # 1日前にして確実に検索に引っかかるようにする
    dt = dt - datetime.timedelta(days=1)

    return f"{dt.isoformat()}.0000+00:00"


# @retry(
#     exceptions=(RequestException),
#     # Requests per 15-minute window unless otherwise stated
#     delay=RATE_LIMITS_SLEEP,
# )
def search_tweet(
    query_params: Dict[str, Any],
    bearer_token: str,
    search_url: str = "https://api.twitter.com/2/tweets/search/all",
) -> Optional[SearchTweetResult]:

    # Full-archive has a 1 request / 1 second limit
    time.sleep(1)

    res = requests.request(
        method="GET",
        url=search_url,
        auth=partial(bearer_oauth, bearer_token=bearer_token),
        params=query_params,
    )
    if res.status_code != 200:
        # raise Exception(res.status_code, res.text, query_params)
        return None

    res_json = res.json()

    search_tweet_meta = SearchTweetMeta(
        newest_id=res_json["meta"].get("newest_id"),
        oldest_id=res_json["meta"].get("oldest_id"),
        result_count=res_json["meta"]["result_count"],
    )
    if search_tweet_meta.result_count < 1:
        return None

    search_tweet_data_list = [
        SearchTweetData(
            created_at=d["created_at"],
            text=d["text"],
            tweet_id=d["id"],
            author_id=d["author_id"],
        )
        for d in res_json["data"]
    ]
    search_tweet_includes = SearchTweetIncludes(
        users=[
            SearchTweetInclude(user_id=u["id"], name=u["name"], username=u["username"])
            for u in res_json["includes"]["users"]
        ]
    )
    return SearchTweetResult(
        data=search_tweet_data_list,
        includes=search_tweet_includes,
        meta=search_tweet_meta,
    )


def search_around_n_hours_tweets_with_username(
    iso_str_datetime: str, username: str, bearer_token: str, n_hours: int = 1
) -> Optional[SearchTweetResult]:

    base_datetime = datetime.datetime.strptime(
        iso_str_datetime, "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    dt_start = base_datetime - datetime.timedelta(hours=n_hours)
    dt_end = base_datetime + datetime.timedelta(hours=n_hours)

    query_params = {
        "query": f"from:{username}",
        "tweet.fields": "created_at",
        "expansions": "author_id",
        "start_time": f"{dt_start.isoformat()}.0000+00:00",
        "end_time": f"{dt_end.isoformat()}.0000+00:00",
    }
    search_tweet_result = search_tweet(
        query_params=query_params, bearer_token=bearer_token
    )

    return search_tweet_result


def collect_additional_tweet(
    sentence: str, tweet_datetime: str, bearer_token: str
) -> Optional[SearchTweetResult]:

    sentence = clean_text(sentence)
    tweet_datetime_for_search = get_tweet_datetime(tweet_datetime)
    if tweet_datetime_for_search is None:
        return None

    query_params = {
        "query": sentence,
        "tweet.fields": "created_at",
        "expansions": "author_id",
        "start_time": tweet_datetime_for_search,
    }

    search_tweet_result = search_tweet(
        query_params=query_params, bearer_token=bearer_token
    )
    if search_tweet_result is None:
        return None

    if search_tweet_result.meta.result_count < 1:
        return None

    iso_str_datetime = search_tweet_result.data[0].created_at
    username = search_tweet_result.includes.users[0].username

    search_tweet_result = search_around_n_hours_tweets_with_username(
        iso_str_datetime=iso_str_datetime,
        username=username,
        bearer_token=bearer_token,
    )

    return search_tweet_result


def collect_additional_tweets(
    wrime_tsv_path: pathlib.Path,
    extended_wrime_dir: pathlib.Path,
):

    df = pd.read_csv(wrime_tsv_path, sep="\t")
    bearer_token = get_bearer_token()

    req_num = 0

    with tqdm(range(len(df)), ncols=100) as pbar:

        for i in pbar:

            if req_num != 0 and req_num % 300 == 0:
                for sec_sleep in range(RATE_LIMITS_SLEEP):
                    time.sleep(1)
                    pbar.set_description(
                        f"[Now sleeping: {datetime.timedelta(seconds=RATE_LIMITS_SLEEP-sec_sleep)}]"
                    )

            pbar.set_description("")  # reset the description of the progress bar

            data = df.iloc[i]
            user_id = data["UserID"]

            user_dir = extended_wrime_dir / f"user_id_{user_id:02d}"
            user_dir.mkdir(exist_ok=True, parents=True)

            sentence = data["Sentence"]
            tweet_datetime = data["Datetime"]

            md5_hash = hashlib.md5(f"{sentence}_{tweet_datetime}".encode()).hexdigest()
            json_file = user_dir / f"{md5_hash}.json"

            if json_file.exists():
                continue

            search_tweet_result = collect_additional_tweet(
                bearer_token=bearer_token,
                sentence=sentence,
                tweet_datetime=tweet_datetime,
            )
            req_num += 1

            if search_tweet_result is not None:
                search_tweet_result_dict = asdict(search_tweet_result)

                search_tweet_result_dict["Sentence"] = sentence
                search_tweet_result_dict["UserID"] = int(user_id)

                with json_file.open("w") as wf:
                    json.dump(search_tweet_result_dict, wf, ensure_ascii=False)


def parse_args(prog: Optional[str] = None) -> argparse.Namespace:

    parser = argparse.ArgumentParser(prog=prog, description="追加でツイートを集める君")
    parser.add_argument(
        "--wrime-tsv",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "data" / "wrime.tsv",
        help="path to `wrime.tsv`",
    )
    parser.add_argument(
        "--extended-wrime-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent / "data" / "extended_wrime/",
        help="path to `extended_wrime.jsonl`",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    collect_additional_tweets(
        wrime_tsv_path=args.wrime_tsv, extended_wrime_dir=args.extended_wrime_dir
    )


if __name__ == "__main__":
    main()
