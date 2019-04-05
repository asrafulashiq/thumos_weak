# check max len
python main_detect.py -m len200 --max-seqlen 200 > ./tmp_log/len200.txt
python main_detect.py -m len300 --max-seqlen 300 > ./tmp_log/len300.txt
python main_detect.py -m len500 --max-seqlen 500 > ./tmp_log/len500.txt
python main_detect.py -m len50 --max-seqlen 50 > ./tmp_log/len50.txt

# check lambda
python main_detect.py -m lamb7 --Lambda 0.7 > ./tmp_log/lamb7.txt
python main_detect.py -m lamb8 --Lambda 0.8 > ./tmp_log/lamb8.txt
python main_detect.py -m lamb3 --Lambda 0.3 > ./tmp_log/lamb3.txt

# check beta1
python main_detect.py -m b1_10 --beta1 10 > ./tmp_log/b1_10.txt
python main_detect.py -m b1_1 --beta1 1 > ./tmp_log/b1_1.txt
python main_detect.py -m b1_1000 --beta1 1000 > ./tmp_log/b1_1000.txt

# check beta2
python main_detect.py -m b2_10 --beta2 10 > ./tmp_log/b2_10.txt
python main_detect.py -m b2_1 --beta2 1 > ./tmp_log/b2_1.txt
python main_detect.py -m b2_100 --beta2 100 > ./tmp_log/b2_100.txt

# check similar size
python main_detect.py -m ssize10 --similar-size 10 --max-iter 8000 > ./tmp/ssize10.txt
python main_detect.py -m ssize20 --similar-size 20 --max-iter 5000 > ./tmp/ssize20.txt

# check num-similar
python main_detect.py -m nsim10 --num-similar 10 --max-iter 8000 > ./tmp/nsim10.txt
python main_detect.py -m nsim10 --num-similar 20 --max-iter 5000 > ./tmp/nsim20.txt
python main_detect.py -m nsim3 --num-similar 3 --max-iter 15000 > ./tmp/nsim3.txt


