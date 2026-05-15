                     precision    recall  f1-score   support

               wheat_Aphid       0.15      0.19      0.17       229
          wheat_Black Rust       0.40      0.85      0.55       165
               wheat_Blast       0.00      0.00      0.00       265
          wheat_Brown Rust       0.80      0.30      0.44       145
     wheat_Common Root Rot       0.26      0.75      0.39       127
wheat_Fusarium Head Blight       0.24      0.46      0.32       105
             wheat_Healthy       0.36      0.27      0.31       608
         wheat_Leaf Blight       0.55      0.32      0.41       337
              wheat_Mildew       0.42      0.92      0.58       233
                wheat_Mite       0.56      0.46      0.50       160
            wheat_Septoria       0.84      0.75      0.79       410
                wheat_Smut       0.72      0.41      0.52       266
            wheat_Stem Fly       0.24      0.58      0.34        33
            wheat_Tan Spot       0.01      0.01      0.01       151
         wheat_Yellow Rust       0.30      0.24      0.27       359
rice_Bacterial Leaf Blight       0.98      0.69      0.81       134
           rice_Brown Spot       0.85      0.20      0.32       116
    rice_Healthy Rice Leaf       0.78      0.11      0.20       125
           rice_Leaf Blast       0.32      0.40      0.35        60
           rice_Leaf Scald       0.38      1.00      0.55        65
        rice_Sheath Blight       0.45      0.62      0.52        79
         sugarcane_Healthy       0.78      0.26      0.39        69
          sugarcane_Mosaic       0.42      1.00      0.60        39
          sugarcane_Redrot       0.65      0.69      0.67        99
            sugarcane_Rust       0.95      1.00      0.97        38
  sugarcane_Yellow Disease       0.00      0.00      0.00         0
   cotton_Bacterial Blight       0.00      0.00      0.00         0
cotton_Leaf Hopper Jassids       0.00      0.00      0.00         0

                  accuracy                           0.43      4417
                 macro avg       0.44      0.45      0.39      4417
              weighted avg       0.47      0.43      0.41      4417


  File "D:\clg\major project\Project folder\projectv2\train_multimodal.py", line 454, in <module>
    main()
  File "D:\clg\major project\Project folder\projectv2\train_multimodal.py", line 439, in main
    print(classification_report(crop_labels_remap, crop_preds_remap, target_names=target_names))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\K Kishore Kumar\AppData\Roaming\Python\Python312\site-packages\sklearn\utils\_param_validation.py", line 218, in wrapper   
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\K Kishore Kumar\AppData\Roaming\Python\Python312\site-packages\sklearn\metrics\_classification.py", line 3097, in classification_report
    raise ValueError(
ValueError: Number of classes, 16, does not match size of target_names, 15. Try specifying the labels parameter
PS D:\clg\major project\Project folder\projectv2>
 *  History restored 

PS D:\clg\major project\Project folder> python train_multimodal.py --variant multimodal --warmup 5 --epochs 50
>> 
C:\Python312\python.exe: can't open file 'D:\\clg\\major project\\Project folder\\train_multimodal.py': [Errno 2] No such file or directory
PS D:\clg\major project\Project folder> cd projectv2
PS D:\clg\major project\Project folder\projectv2> python train_multimodal.py --variant multimodal --warmup 5 --epochs 50
>> 
Using device: cuda

PHASE 1: Warmup (5 epochs)

Warmup Epoch 1/5
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [03:04<00:00,  2.69it/s, loss=2.29, acc=24.5]
  Train Loss: 2.8906 | Acc: 0.2448
  Val Loss:   2.4792 | Acc: 0.4887

Warmup Epoch 2/5
Training: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [02:31<00:00,  3.26it/s, loss=2.11, acc=44] 
  Train Loss: 2.3871 | Acc: 0.4399
  Val Loss:   2.1284 | Acc: 0.6271

Warmup Epoch 3/5
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:23<00:00,  3.44it/s, loss=1.32, acc=54.9] 
  Train Loss: 2.1100 | Acc: 0.5492
  Val Loss:   1.8792 | Acc: 0.7143

Warmup Epoch 4/5
Training: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [02:16<00:00,  3.63it/s, loss=3, acc=61.2] 
  Train Loss: 1.9585 | Acc: 0.6116
  Val Loss:   1.7490 | Acc: 0.7652

Warmup Epoch 5/5
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:17<00:00,  3.59it/s, loss=2.73, acc=63.1] 
  Train Loss: 1.9408 | Acc: 0.6314
  Val Loss:   1.6561 | Acc: 0.7941

PHASE 2: Fine-Tuning (50 epochs)
Unfroze top 2/6 blocks (18,355,912 params trainable)

Fine-Tune Epoch 1/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.33it/s, loss=1.75, acc=70.6]
  Train Loss: 1.7598 | Acc: 0.7065
  Val Loss:   1.4308 | Acc: 0.8946
  Best model saved! (Acc: 0.8946)

Fine-Tune Epoch 2/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:26<00:00,  3.39it/s, loss=1.19, acc=79.6] 
  Train Loss: 1.5774 | Acc: 0.7961
  Val Loss:   1.3266 | Acc: 0.9334
  Best model saved! (Acc: 0.9334)

Fine-Tune Epoch 3/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.33it/s, loss=1.27, acc=84.1] 
  Train Loss: 1.4603 | Acc: 0.8412
  Val Loss:   1.2490 | Acc: 0.9567
  Best model saved! (Acc: 0.9567)

Fine-Tune Epoch 4/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.33it/s, loss=1.29, acc=85.8] 
  Train Loss: 1.4248 | Acc: 0.8576
  Val Loss:   1.2061 | Acc: 0.9694
  Best model saved! (Acc: 0.9694)

Fine-Tune Epoch 5/50
Training: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.33it/s, loss=1.69, acc=86] 
  Train Loss: 1.4245 | Acc: 0.8604
  Val Loss:   1.1863 | Acc: 0.9767
  Best model saved! (Acc: 0.9767)

Fine-Tune Epoch 6/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.34it/s, loss=1.12, acc=85.7] 
  Train Loss: 1.4411 | Acc: 0.8567
  Val Loss:   1.1693 | Acc: 0.9809
  Best model saved! (Acc: 0.9809)

Fine-Tune Epoch 7/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:24<00:00,  3.42it/s, loss=1.14, acc=86.5] 
  Train Loss: 1.4148 | Acc: 0.8646
  Val Loss:   1.1672 | Acc: 0.9817
  Best model saved! (Acc: 0.9817)

Fine-Tune Epoch 8/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:26<00:00,  3.38it/s, loss=1.09, acc=87.8] 
  Train Loss: 1.3789 | Acc: 0.8781
  Val Loss:   1.1611 | Acc: 0.9829
  Best model saved! (Acc: 0.9829)

Fine-Tune Epoch 9/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:20<00:00,  3.52it/s, loss=1.09, acc=87.7] 
  Train Loss: 1.3866 | Acc: 0.8769
  Val Loss:   1.1584 | Acc: 0.9826

Fine-Tune Epoch 10/50
Training: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [02:18<00:00,  3.57it/s, loss=1.02, acc=86] 
  Train Loss: 1.4211 | Acc: 0.8604
  Val Loss:   1.1620 | Acc: 0.9823

Fine-Tune Epoch 11/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:28<00:00,  2.37it/s, loss=0.867, acc=87.4] 
  Train Loss: 1.3875 | Acc: 0.8738
  Val Loss:   1.1715 | Acc: 0.9809

Fine-Tune Epoch 12/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:17<00:00,  3.60it/s, loss=2.01, acc=86.1] 
  Train Loss: 1.4265 | Acc: 0.8610
  Val Loss:   1.1575 | Acc: 0.9817

Fine-Tune Epoch 13/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [03:22<00:00,  2.45it/s, loss=1.18, acc=88.3] 
  Train Loss: 1.3639 | Acc: 0.8826
  Val Loss:   1.1464 | Acc: 0.9853
  Best model saved! (Acc: 0.9853)

Fine-Tune Epoch 14/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:26<00:00,  2.39it/s, loss=0.909, acc=87.7] 
  Train Loss: 1.3776 | Acc: 0.8774
  Val Loss:   1.1370 | Acc: 0.9865
  Best model saved! (Acc: 0.9865)

Fine-Tune Epoch 15/50
Training: 100%|██████████████████████████████████████████████████████████████████████| 495/495 [03:44<00:00,  2.20it/s, loss=2.4, acc=88.9] 
  Train Loss: 1.3342 | Acc: 0.8889
  Val Loss:   1.1395 | Acc: 0.9862

Fine-Tune Epoch 16/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [03:59<00:00,  2.07it/s, loss=0.96, acc=87.9] 
  Train Loss: 1.3579 | Acc: 0.8789
  Val Loss:   1.1466 | Acc: 0.9841

Fine-Tune Epoch 17/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:20<00:00,  3.51it/s, loss=1.02, acc=87.8] 
  Train Loss: 1.3678 | Acc: 0.8775
  Val Loss:   1.1412 | Acc: 0.9841

Fine-Tune Epoch 18/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.33it/s, loss=0.925, acc=87.9] 
  Train Loss: 1.3657 | Acc: 0.8785
  Val Loss:   1.1291 | Acc: 0.9856

Fine-Tune Epoch 19/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:00<00:00,  2.74it/s, loss=0.859, acc=87.8] 
  Train Loss: 1.3590 | Acc: 0.8783
  Val Loss:   1.1240 | Acc: 0.9876
  Best model saved! (Acc: 0.9876)

Fine-Tune Epoch 20/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:18<00:00,  2.50it/s, loss=0.823, acc=88.1]
  Train Loss: 1.3495 | Acc: 0.8813
  Val Loss:   1.1197 | Acc: 0.9882
  Best model saved! (Acc: 0.9882)

Fine-Tune Epoch 21/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [02:53<00:00,  2.85it/s, loss=0.687, acc=90.4] 
  Train Loss: 1.2810 | Acc: 0.9039
  Val Loss:   1.1136 | Acc: 0.9894
  Best model saved! (Acc: 0.9894)

Fine-Tune Epoch 22/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [03:49<00:00,  2.16it/s, loss=1.14, acc=87.9] 
  Train Loss: 1.3414 | Acc: 0.8794
  Val Loss:   1.1130 | Acc: 0.9894

Fine-Tune Epoch 23/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:45<00:00,  2.20it/s, loss=0.833, acc=87.6] 
  Train Loss: 1.3525 | Acc: 0.8765
  Val Loss:   1.1111 | Acc: 0.9900
  Best model saved! (Acc: 0.9900)

Fine-Tune Epoch 24/50
Training: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [03:36<00:00,  2.28it/s, loss=1.16, acc=88] 
  Train Loss: 1.3346 | Acc: 0.8799
  Val Loss:   1.1137 | Acc: 0.9897

Fine-Tune Epoch 25/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:56<00:00,  2.09it/s, loss=0.762, acc=87.6] 
  Train Loss: 1.3520 | Acc: 0.8760
  Val Loss:   1.1153 | Acc: 0.9897

Fine-Tune Epoch 26/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [03:40<00:00,  2.24it/s, loss=1.67, acc=87.5] 
  Train Loss: 1.3605 | Acc: 0.8754
  Val Loss:   1.1131 | Acc: 0.9894

Fine-Tune Epoch 27/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [03:44<00:00,  2.20it/s, loss=0.843, acc=89.3] 
  Train Loss: 1.2935 | Acc: 0.8932
  Val Loss:   1.1092 | Acc: 0.9894

Fine-Tune Epoch 28/50
Training: 100%|████████████████████████████████████████████████████████████████████| 495/495 [02:18<00:00,  3.58it/s, loss=0.891, acc=88.7] 
  Train Loss: 1.3258 | Acc: 0.8866
  Val Loss:   1.1118 | Acc: 0.9900

Fine-Tune Epoch 29/50
Training: 100%|████████████████████████████████████████████████████████████████████████| 495/495 [02:28<00:00,  3.34it/s, loss=1.4, acc=90] 
  Train Loss: 1.2752 | Acc: 0.9005
  Val Loss:   1.1085 | Acc: 0.9900

Fine-Tune Epoch 30/50
Training: 100%|█████████████████████████████████████████████████████████████████████| 495/495 [02:42<00:00,  3.04it/s, loss=2.49, acc=89.4] 
  Train Loss: 1.2882 | Acc: 0.8940
  Val Loss:   1.1090 | Acc: 0.9900

Fine-Tune Epoch 31/50
Training: 100%|███████████████████████████████████████████████████████████████████████| 495/495 [02:37<00:00,  3.14it/s, loss=2.37, acc=90] 
  Train Loss: 1.2785 | Acc: 0.9004
  Val Loss:   1.1216 | Acc: 0.9888
Early stopping at epoch 31

Training finished. Evaluating on Test Set...

Classification Report:
                            precision    recall  f1-score   support

               wheat_Aphid       0.99      0.97      0.98       136
          wheat_Black Rust       0.99      0.99      0.99        87
               wheat_Blast       1.00      1.00      1.00        98
          wheat_Brown Rust       1.00      1.00      1.00       191
     wheat_Common Root Rot       0.98      0.98      0.98        93
wheat_Fusarium Head Blight       0.98      0.99      0.98        92
             wheat_Healthy       1.00      1.00      1.00       308
         wheat_Leaf Blight       0.99      1.00      1.00       127
              wheat_Mildew       0.98      0.99      0.98       163
                wheat_Mite       1.00      0.98      0.99       120
            wheat_Septoria       1.00      1.00      1.00       172
                wheat_Smut       0.99      0.99      0.99       197
            wheat_Stem Fly       0.89      0.94      0.92        36
            wheat_Tan Spot       0.99      1.00      1.00       116
         wheat_Yellow Rust       1.00      0.99      1.00       196
rice_Bacterial Leaf Blight       1.00      1.00      1.00        96
           rice_Brown Spot       0.99      0.99      0.99        97
    rice_Healthy Rice Leaf       1.00      1.00      1.00        98
           rice_Leaf Blast       1.00      0.98      0.99        96
           rice_Leaf Scald       0.96      0.99      0.97        95
        rice_Sheath Blight       1.00      0.99      0.99        95
         sugarcane_Healthy       0.97      0.99      0.98        79
          sugarcane_Mosaic       1.00      1.00      1.00        70
          sugarcane_Redrot       0.99      1.00      0.99        78
            sugarcane_Rust       1.00      0.96      0.98        78
  sugarcane_Yellow Disease       0.99      1.00      0.99        76
   cotton_Bacterial Blight       0.95      0.95      0.95        38
         cotton_Curl Virus       0.98      0.98      0.98        65
       cotton_Healthy Leaf       0.95      1.00      0.97        39
   cotton_Herbicide Damage       1.00      0.98      0.99        42
cotton_Leaf Hopper Jassids       1.00      0.97      0.99        34
     cotton_Leaf Reddening       0.99      0.99      0.99        87
   cotton_Leaf Variegation       1.00      1.00      1.00        18

                  accuracy                           0.99      3413
                 macro avg       0.99      0.99      0.99      3413
              weighted avg       0.99      0.99      0.99      3413


--- WHEAT Classification Report ---
                            precision    recall  f1-score   support

               wheat_Aphid       0.99      0.97      0.98       136
          wheat_Black Rust       0.99      0.99      0.99        87
               wheat_Blast       1.00      1.00      1.00        98
          wheat_Brown Rust       1.00      1.00      1.00       191
     wheat_Common Root Rot       0.98      0.98      0.98        93
wheat_Fusarium Head Blight       0.98      0.99      0.98        92
             wheat_Healthy       1.00      1.00      1.00       308
         wheat_Leaf Blight       0.99      1.00      1.00       127
              wheat_Mildew       0.98      0.99      0.98       163
                wheat_Mite       1.00      0.98      0.99       120
            wheat_Septoria       1.00      1.00      1.00       172
                wheat_Smut       0.99      0.99      0.99       197
            wheat_Stem Fly       0.89      0.94      0.92        36
            wheat_Tan Spot       0.99      1.00      1.00       116
         wheat_Yellow Rust       1.00      0.99      1.00       196

                  accuracy                           0.99      2132
                 macro avg       0.99      0.99      0.99      2132
              weighted avg       0.99      0.99      0.99      2132


--- RICE Classification Report ---
                            precision    recall  f1-score   support

rice_Bacterial Leaf Blight       1.00      1.00      1.00        96
           rice_Brown Spot       0.99      0.99      0.99        97
    rice_Healthy Rice Leaf       1.00      1.00      1.00        98
           rice_Leaf Blast       1.00      0.98      0.99        96
           rice_Leaf Scald       0.96      0.99      0.97        95
        rice_Sheath Blight       1.00      0.99      0.99        95

                  accuracy                           0.99       577
                 macro avg       0.99      0.99      0.99       577
              weighted avg       0.99      0.99      0.99       577


--- SUGARCANE Classification Report ---
                          precision    recall  f1-score   support

       sugarcane_Healthy       0.97      0.99      0.98        79
        sugarcane_Mosaic       1.00      1.00      1.00        70
        sugarcane_Redrot       0.99      1.00      0.99        78
          sugarcane_Rust       1.00      0.96      0.98        78
sugarcane_Yellow Disease       0.99      1.00      0.99        76

                accuracy                           0.99       381
               macro avg       0.99      0.99      0.99       381
            weighted avg       0.99      0.99      0.99       381


--- COTTON Classification Report ---
                            precision    recall  f1-score   support

   cotton_Bacterial Blight       0.95      0.95      0.95        38
         cotton_Curl Virus       0.98      0.98      0.98        65
       cotton_Healthy Leaf       0.95      1.00      0.97        39
   cotton_Herbicide Damage       1.00      0.98      0.99        42
cotton_Leaf Hopper Jassids       1.00      0.97      0.99        34
     cotton_Leaf Reddening       0.99      0.99      0.99        87
   cotton_Leaf Variegation       1.00      1.00      1.00        18

                  accuracy                           0.98       323
                 macro avg       0.98      0.98      0.98       323
              weighted avg       0.98      0.98      0.98       323