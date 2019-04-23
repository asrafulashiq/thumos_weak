


## effect of l_w (topk)
# python main_detect.py -m test --max-iter 7001 --topk 200 --topk2 20 >> ./tmp_log/top.txt

# python main_detect.py -m test --max-iter 7001 --topk 100 --topk2 10 >> ./tmp_log/top.txt

# python main_detect.py -m test --max-iter 7001 --topk 20 --topk2 5 >> ./tmp_log/top.txt


# ### effect of dis
# python   main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 4 --dis 1 >> ./tmp_log/t1
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 4 --dis 2 >> ./tmp_log/t1
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 4 --dis 3 >> ./tmp_log/t1
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 4 --dis 3.5 >> ./tmp_log/t1

# ### effect of clip
# python  main_detect.py -m thumos_base --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 5 --dis 3 >> ./tmp_log/t2
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 4 --dis 3 >> ./tmp_log/t2
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 3 --dis 3 >> ./tmp_log/t2
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 7000 --topk 60 --topk2 10 --clip 2 --dis 3 >> ./tmp_log/t2

# ### effect of batch size
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 4000 --num-similar 10 --similar-size 4 --lr 0.0004 --dis 3 >> ./tmp_log/t3
# python  main_detect.py -m thumos_basetest --Lambda 0.5 --max-iter \
#                 4000 --num-similar 4 --similar-size 10 --lr 0.0004 --dis 3 >> ./tmp_log/t3