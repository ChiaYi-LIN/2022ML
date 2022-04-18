rm -rf ./r10922124_hw6
mkdir -p ./r10922124_hw6

cp ./hw6_stylegan.py ./r10922124_hw6
cp ./plot_grad_norm.py ./r10922124_hw6
cp ./README.md ./r10922124_hw6

rm -f ./r10922124_hw6.zip
zip -r ./r10922124_hw6.zip ./r10922124_hw6

rm -rf ./r10922124_hw6