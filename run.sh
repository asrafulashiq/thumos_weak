echo 1
python main_detect.py -m tmp1 --dis 0.8 --max-iter 5000 > tmp_log/tmp1.txt
echo 2
python main_detect.py -m tmp2 --dis 1 --max-iter 5000 > tmp_log/tmp2.txt
echo 3
python main_detect.py -m tmp3 --dis 2 --max-iter 5000 > tmp_log/tmp3.txt
echo 4
python main_detect.py -m tmp4 --dis 0.3 --max-iter 5000 > tmp_log/tmp4.txt
