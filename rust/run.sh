#!/bin/bash
set -e

TMP_FOLDER=./.twembeddings

# Cleanup
rm -rf $TMP_FOLDER
mkdir $TMP_FOLDER

# Building binaries
cargo build --release
echo

TWEMBEDDINGS="./target/release/twembeddings"
TOTAL=`xsv count $1`

echo "1. Extracting vocabulary"
echo "------------------------"
$TWEMBEDDINGS vocab $1 --total $TOTAL --tsv > $TMP_FOLDER/vocab.csv
echo

echo "2. Determining window size"
echo "--------------------------"
WINDOW=`$TWEMBEDDINGS window $1 --raw --total $TOTAL --tsv`
echo "Optimal window size should be: $WINDOW"
echo

echo "3. Applying clustering algorithm"
echo "--------------------------------"
$TWEMBEDDINGS nn $TMP_FOLDER/vocab.csv $1 -w $WINDOW --total $TOTAL --tsv --threshold 0.6 > $TMP_FOLDER/nn.csv
echo

echo "4. Evaluating"
echo "-------------"
$TWEMBEDDINGS eval $2 $TMP_FOLDER/nn.csv --total $TOTAL
