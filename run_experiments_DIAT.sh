#!/bin/bash



# Array of d_state values to test
hidden_size_values_radmamba=(8 16 32 64 80)
hidden_size_values_resnet=(5 7 12 20 30)
hidden_size_values_cnnlstm=(2 4 6 16 28 34 46)
hidden_size_values_bilstm=(1 2 4 10 20 28 34)
hidden_size_values_cnngru=(1 2 4 6 16 28)



# # Run for 10 different random seeds (0-9)
for seed in {0..9}; do
    for hidden_size in "${hidden_size_values_radmamba[@]}"; do
        echo "Running experiment with seed=$seed, hidden_size=$hidden_size"
        python main.py \
            --dataset_name 'DIAT' \
            --seed $seed \
            --Classification_hidden_size $hidden_size \
            --Classification_backbone 'radmamba' \
            --dim $hidden_size 
    done
    for hidden_size in "${hidden_size_values_cnnlstm[@]}"; do
        echo "Running experiment with seed=$seed, hidden_size=$hidden_size"
        python main.py \
            --dataset_name 'DIAT' \
            --seed $seed \
            --Classification_hidden_size $hidden_size \
            --Classification_backbone 'cnnlstm' \
            --lr 1e-4
    done
    for hidden_size in "${hidden_size_values_bilstm[@]}"; do
        echo "Running experiment with seed=$seed, hidden_size=$hidden_size"
        python main.py \
            --dataset_name 'DIAT' \
            --seed $seed \
            --Classification_hidden_size $hidden_size \
            --Classification_backbone 'bilstm' \
            --lr 1e-4
    done
        for hidden_size in "${hidden_size_values_cnngru[@]}"; do
        echo "Running experiment with seed=$seed, hidden_size=$hidden_size"
        python main.py \
            --dataset_name 'DIAT' \
            --seed $seed \
            --Classification_hidden_size $hidden_size \
            --Classification_backbone 'cnngru' \
            --lr 1e-4
    done
    for hidden_size in "${hidden_size_values_resnet[@]}"; do
        echo "Running experiment with seed=$seed, hidden_size=$hidden_size"
        python main.py \
            --dataset_name 'DIAT' \
            --seed $seed \
            --Classification_hidden_size $hidden_size \
            --Classification_backbone 'resnet' \
            --lr 1e-4
    done
done 



