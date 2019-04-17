
python -m pdb -c c  main_detect.py -m test_new --Lambda 1 --max-iter 2500 --topk 40 --topk2 5 >> ./tmp_log/test.txt
python -m pdb -c c  main_detect.py -m test_new --Lambda 1 --max-iter 2500 --topk 100 --topk2 12 >> ./tmp_log/test.txt
python -m pdb -c c  main_detect.py -m test_new --Lambda 1 --max-iter 2500 --topk 20 --topk2 3 >> ./tmp_log/test.txt
python -m pdb -c c  main_detect.py -m test_new --Lambda 1 --max-iter 2500 --topk 80 --topk2 10 >> ./tmp_log/test.txt


