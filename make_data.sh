cd data

# Download wrime dataset
curl -O https://raw.githubusercontent.com/ids-cv/wrime/master/wrime.tsv
# Fix column name
if sed --version > /dev/null 2>&1; then
  # Linux (GNU)
  sed -i -e "1s/Saddness/Sadness/g" wrime.tsv
else
  # Mac (BSD)
  sed -i "" -e "1s/Saddness/Sadness/g" wrime.tsv
fi
echo
