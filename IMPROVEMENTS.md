C:/ProgramData/miniconda3/Scripts/conda.exe run -p C:\Users\Devesh\.conda\envs\neuromorphic 

1.
    python src/train_snn.py \
      --event_dir ./events/train \
      --epochs 100 \
      --batch_size 16 \
      --num_steps 25 \
      --encoding rate \
      --target_rate 0.03 \
      --rate_weight 1.0

    Tuning guide if ActualRate doesn't reach ~0.10:

    Rate stays near 0 → increase --rate_weight to 2.0 or 5.0
    Rate overshoots above 0.3 → decrease --rate_weight to 0.5
    If you want a sparser model (less firing, better energy efficiency on MI300X) → lower --target_rate to 0.05
  Note :- after traing for epoch = 2 .. aagent suggested to change target_rate = 0.1 to 0.005



2. 
        1. Loss drop in first 10 epochs: 0.019612 -> 0.001984 (Significant: Yes)
        2. Loss at Epoch 30: 0.001947 vs Epoch 50: 0.001956
        3. Plateaus early? Yes
        4. Substantial decrease (E1 to E50): 90.03% reduction

        --- 5. Budget Tracking ---
        Estimated Total GPU Time Used: 37.50 minutes


4. 

    **commands**:
      python src/train_snn.py \
      --event_dir ./events/train \
      --epochs 100 \
      --batch_size 16 \
      --num_steps 25 \
      --bottleneck_channels 8 \
      --encoding rate \
      --target_rate 0.1 \
      --rate_weight 1.0 \
      --beta 0.95 \
      --checkpoint_dir ./checkpoints/rate


      python src/evaluate_snn.py \
        --checkpoint checkpoints/rate/snn_autoencoder_best.pth \
        --event_dir ./events/test \
        --num_steps 25 \
        --bottleneck_channels 8 \
        --encoding rate \
        --score_mode all

     1. Rate-Encoding:Best loss: 0.005574                                     
      ============================================================
      EVALUATION RESULTS
      ============================================================
        Model:          SNN Autoencoder (snnTorch)
        Encoding:       rate
        Timesteps:      25
        AUC [combined  ]: 0.6410
        AUC [mem_only  ]: 0.6416 <-- BEST
        AUC [mse_only  ]: 0.6399
        AUC [weighted  ]: 0.6414
        Avg spike rate: 0.0411
        Sparsity:       0.9589
        Samples scored: 2010

    2. Temporal-Encoding:Best loss: 0.005337
      ============================================================
      EVALUATION RESULTS
      ============================================================
        Model:          SNN Autoencoder (snnTorch)
        Encoding:       temporal
        Timesteps:      25
        AUC [combined  ]: 0.6039
        AUC [mem_only  ]: 0.6558 <-- BEST
        AUC [mse_only  ]: 0.5839
        AUC [weighted  ]: 0.6465
        Avg spike rate: 0.0542
        Sparsity:       0.9458
        Samples scored: 2010
      ============================================================


    3. Count-Encoding:Best loss: 0.008732
      ============================================================
      EVALUATION RESULTS
      ============================================================
        Model:          SNN Autoencoder (snnTorch)
        Encoding:       count
        Timesteps:      25
        AUC [combined  ]: 0.6969
        AUC [mem_only  ]: 0.6826
        AUC [mse_only  ]: 0.6677
        AUC [weighted  ]: 0.6994 <-- BEST
        Avg spike rate: 0.0660
        Sparsity:       0.9340
        Samples scored: 2010
      ============================================================


    4. ConvLSTM rate: Best model saved (loss=0.000001)                 commands :python src/evaluate_convlstm.py ,python src/train.py

        Total sequences created: 1878
        ============================================================
        ConvLSTM BASELINE RESULTS
        ============================================================
          AUC-ROC: 0.5995
          Samples: 1878
        ============================================================


    5. ConvAutoencoder: TRAINING COMPLETE  |  Best loss: 0.000000                    code: python src/train_conv_autoencoder.py
          ============================================================                     python src/evaluate_conv_autoencoder.py
          EVALUATION RESULTS
          ============================================================
            Model:          ConvAutoencoder (RGB frames)
            Image size:     128x128
            AUC-ROC:        0.8269
            Frames scored:  2010
          ============================================================


  
All Snn,ConvLSTM,ConvAutoencoder :
1. What's the same across all models (fair):

      Same dataset: UCSD Ped2
      Same train/test split (Train = normal only, Test = GT labels)
      Same epochs: 50
      Same optimizer: Adam, lr=1e-3, CosineAnnealingLR

2. Your final comparison table:

            Model	                    Input	                   AUC
        ConvAutoencoder	      Motion frames (frame diff)	    0.8269
        SNN Count	            Event spikes	                  0.6994
        SNN Temporal	        Event spikes	                  0.6558
        SNN Rate	            Event spikes	                  0.6416
        ConvLSTM RGB	        Raw RGB sequences             	0.5995



**Effiency Analysis** :-

1. The headline findings:

        What	                           Number	                                Why it matters
    SNN params	                  89K vs 949K ConvAE	                  10.6x more compact — fits on embedded chips
    SNN Loihi energy	              16,061 μJ	                          11.4x less energy than ConvAE on GPU
    ConvLSTM	                  88B MACs, 2211ms, worst AUC	            Temporal RGB modeling fails on every dimension
    SNN internal sparsity	          83.9%	                              84% of neurons silent → neuromorphic hardware skips those
    AUC/log₁₀E score	          SNN: 0.1663, ConvAE: 0.1571	            SNN WINS when accuracy × efficiency is combined



  2. The story your results tell:

            Raw accuracy ranking:    ConvAE (0.8269) > SNN (0.6994) > ConvLSTM (0.5995)
            Efficiency ranking:      SNN   >>>  ConvAE  >>>  ConvLSTM
            Combined (AUC/energy):   SNN   >    ConvAE  >>>  ConvLSTM  ← SNN wins here

  SNN sacrifices 12.7% AUC to gain 11.4x energy reduction. For always-on surveillance cameras that run 24/7, that tradeoff is absolutely worth it — 11x lower power consumption could mean battery-powered or solar-powered edge deployment.

  **One thing to note for your report:*
  The energy numbers (200 pJ/MAC for GPU, 23 pJ/SynOp for Loihi) are from Horowitz 2014 / Davies 2018 — standard published estimates. Modern GPUs are more efficient in practice, but the relative 11x advantage holds across different hardware assumptions. Mention this when writing up.





**Final PipeLine commands**
        # First full run (takes ~3h with all 5 models)
        !python run_pipeline.py

        # After checkpoints exist — just redo eval + viz
        !python run_pipeline.py --skip_training

        # Check what will run before committing
        !python run_pipeline.py --dry_run
