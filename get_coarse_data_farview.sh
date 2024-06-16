total_env=5000
chunk_env=100

chunk_num=$total_env/$chunk_env
for ((i=0;i<chunk_num;i++))
do
    echo "chunk idx " $i 
    python ./HW/CV/get_coarse_data_farview.py --num_envs $chunk_env --chunk_idx $i
done
