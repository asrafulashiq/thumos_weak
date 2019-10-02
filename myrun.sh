# effect of num-similar
conda activate torch

python -m pdb main_detect.py --model-name test_euclid --dis 1 --max-iter 5000 >> tmp_log/euclid.txt
python -m pdb main_detect.py --model-name test_euclid --dis 0.5 --max-iter 5000 >> tmp_log/euclid.txt
python -m pdb main_detect.py --model-name test_euclid --dis 3 --max-iter 5000 >> tmp_log/euclid.txt
python -m pdb main_detect.py --model-name test_euclid --dis 10 --max-iter 5000 >> tmp_log/euclid.txt
