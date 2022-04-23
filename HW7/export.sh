rm -rf ./r10922124_hw7
mkdir -p ./r10922124_hw7

cp ./preprocess.py ./r10922124_hw7
cp ./train.sh ./r10922124_hw7
cp ./run_qa.py ./r10922124_hw7
cp ./trainer_qa.py ./r10922124_hw7
cp ./utils_qa.py ./r10922124_hw7
cp ./postprocess.py ./r10922124_hw7
cp ./README.md ./r10922124_hw7

rm -f ./r10922124_hw7.zip
zip -r ./r10922124_hw7.zip ./r10922124_hw7

rm -rf ./r10922124_hw7