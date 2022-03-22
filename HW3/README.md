# Training
在 2022ml_hw3_image_classification.py 中可以經由多次 train 出 5 個 models，分別為
Model_1: resnet18 on default train, valid dataset
Model_2to5: efficientnet_b4 on merged train, valid dataset with 5 folds, each model takes one of the 5 folds as valid set, the rest as train set

# Testing
在 test_ensemble.py 中
有用 test time augmentation 生成 10 個經過 transform 的 test data，與沒有 transform 的 test data (權重 10 倍)
並利用以上 5 個 models 的 predictions 做加總，再用 argmax 產出最後的 prediction