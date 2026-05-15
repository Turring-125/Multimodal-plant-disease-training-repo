# Wheat Model Training Bug - `Healthy` & `Yellow Rust` Confusion

## The Issue
During the PyTorch `EfficientNetV2S` training run, identical symptoms to the old TensorFlow model were observed in the test results. Specifically, the model scored very poorly on the `Healthy` class (e.g., 0.08 recall) and the `Yellow Rust` class (e.g., 0.54 precision). 

While it appeared as though older results were somehow written to the file, these were the true results of the new 5.3-hour PyTorch run. The newly built PyTorch model failed to learn the distinctions because of a data weighting collision.

## Technical Failure Reason
The PyTorch `train_model.py` script mistakenly applied **two** class balancing techniques simultaneously for the Wheat dataset:

1. A `WeightedRandomSampler` applied to the DataLoader (which upsamples minor classes so that each actual training batch is perfectly uniformly balanced).
2. `nn.CrossEntropyLoss(weight=class_weights)` applied to the loss criterion (which gives higher gradients for getting minority classes wrong and heavily reduces penalty for majority classes).

Because the batches were *already* balanced by the sampler, adding the `class_weights` to the loss function effectively "squared" the penalties. This punished the network outrageously when predicting the majority classes (like `Healthy` and `Yellow Rust`), while vastly over-focusing on minority classes (like `Stem fly`). This destroyed the gradient stability and mathematically locked the model from correctly recognizing the largest classes.

## The Fix
The bug was identified and resolved in `train_model.py` on line `277`:
```python
# Before
train_loader = make_loader(train_ds, training=True, use_sampler=True)

# After 
train_loader = make_loader(train_ds, training=True, use_sampler=False)
```

The script is now properly configured to use standard dataset shuffling combined with `class_weights` for the loss function. This perfectly matches the successful rice setup. 

## Next Steps
To resolve the metrics in the classification reports, you will need to simply re-run the script:
```powershell
python train_model.py
```
*Note: Due to the size of the dataset and number of classes, training the Wheat EfficientNetV2S model on the RTX 3060 will take approximately 5 hours.*
