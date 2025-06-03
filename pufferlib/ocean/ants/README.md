```
puffer train puffer_ants --train.device cpu --train.optimizer adam --neptune --neptune-name "matanitah" --neptune-project "ant-sim"
```

```
puffer eval puffer_ants --load-model-path experiments/ANTS-XXX.pt --train.device cpu --train.optimizer adam --neptune
```

```
scripts/build_ocean.sh ants
```