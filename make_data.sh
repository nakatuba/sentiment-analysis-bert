cd data

# Download wrime dataset
curl https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver1.tsv -o wrime.tsv

# Fix column name
if sed --version > /dev/null 2>&1; then
  # Linux (GNU)
  sed -i -e '1s/Saddness/Sadness/g' wrime.tsv
else
  # Mac (BSD)
  sed -i '' -e '1s/Saddness/Sadness/g' wrime.tsv
fi

# Split dataset into train and test
python split_data.py
