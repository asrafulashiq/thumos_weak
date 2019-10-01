# effect of num-similar
conda activate torch

# python main_detect.py -m test_selqn --max-seqlen 50 --max-iter 4000 >> ./tmp_log/len.txt
# python main_detect.py -m test_selqn --max-seqlen 100 --max-iter 4000 >> ./tmp_log/len.txt
python main_detect.py -m test_selqn --max-seqlen 200 --max-iter 4000 >> ./tmp_log/len.txt
python main_detect.py -m test_selqn --max-seqlen 500 --max-iter 4000 >> ./tmp_log/len.txt
python main_detect.py -m test_selqn --max-seqlen 700 --max-iter 4000 >> ./tmp_log/len.txt
python main_detect.py -m test_selqn --max-seqlen 800 --max-iter 4000 >> ./tmp_log/len.txt
