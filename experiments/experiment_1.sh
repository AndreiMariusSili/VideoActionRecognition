set -e

for DATASET in "hmdb" "smth"; do
  for SPLIT in "1"; do
    for CUT in "4q"; do
      for MODEL in "tadn_class" "tarn_class" "i3d_class"; do
        for NUM_FRAMES in "4" "8"; do
          python ../main.py run_experiment --opts="dataset:${DATASET}${SPLIT},cut:${CUT},frames:${NUM_FRAMES},model:${MODEL}"
          python ../main.py eval_experiment --opts="dataset:${DATASET}${SPLIT},cut:${CUT},frames:${NUM_FRAMES},model:${MODEL}"
        done
      done
    done
  done
done
