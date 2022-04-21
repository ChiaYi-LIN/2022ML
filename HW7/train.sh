# python preprocess.py
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true python run_qa.py \
  --model_name_or_path hfl/chinese-roberta-wwm-ext \
  --train_file ./data/all.json \
  --test_file ./data/test.json \
  --do_train True \
  --do_predict True \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./results/roberta_runqa_1e-4/