# Train
```shell
stylegan2_pytorch --data ./faces --name stylegan2 --models_dir ./models --results_dir ./results --image-size 64 --num-train-steps 100000
```

# Generate
```shell
python hw6_stylegan.py
cd stylegan_output
tar -zcf ../stylegan2.tgz ./*.jpg
cd ..
```

# Plot
