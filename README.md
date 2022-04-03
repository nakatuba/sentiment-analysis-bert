# wrime-bert

## Setup Environment

1. Install python modules
```
poetry install
```

2. Download and make data
```
./make_data.sh
```

## Collect additional tweets

- Copy `.env.sample` to `.env` and set `BEARER_TOKEN`
  - The `BEARER_TOKEN` can get from [Twitter API for Academic Research](https://developer.twitter.com/en/products/twitter-api/academic-research)

```shell
cp .env.sample .env
```

- Run the script


```shell
python collect_additional_tweets.py \
    --wrime-tsv data/wrime.tsv \
    --extended-wrime-dir data/extended_wrime
``` 
