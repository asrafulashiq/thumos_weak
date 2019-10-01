# effect of num-similar
conda activate torch

python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
                3000 --num-similar 20 --similar-size 1 --lr 0.0004 --dis 3 >> ./tmp_log/t3.txt



# python main_detect.py -m test_lamb --Lambda 0.1 --max-iter 3000 >> tmp_log/lamb1.txt
# python main_detect.py -m test_lamb --Lambda 0.3 --max-iter 3000 >> tmp_log/lamb1.txt
# python main_detect.py -m test_lamb --Lambda 0.7 --max-iter 3000 >> tmp_log/lamb1.txt
# python main_detect.py -m test_lamb --Lambda 0.9 --max-iter 3000 >> tmp_log/lamb1.txt


