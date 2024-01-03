#!/bin/bash

# 清空之前的輸出文件，如果需要
OUTPUT_PATH="output/output20_30times_PathSampling.txt"
times=2

> "$OUTPUT_PATH" 


# In function
#1
echo "1.57+2.43*X_train" >> "Experiment_1.txt"
for i in $(seq 1 $times)
do
    nohup python main.py "1.57+2.43*X_train" 1 | grep "best fitness" | tail -1 >> "Experiment_1.txt" 2>&1 &
done
#2
echo "X_train**6 + X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train" >> "Experiment_2.txt"" 
for i in $(seq 1 $times)
do
    nohup python main.py X_train**6 + X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train 2 | grep "best fitness" | tail -1 >> "Experiment_2.txt"  2>&1 &
done
#3

echo "X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train" >> "Experiment_3.txt"
for i in $(seq 1 $times)
do
    nohup python main.py X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train 3 | grep "best fitness" | tail -1 >> "Experiment_3.txt" 2>&1 &
done
#4

echo "X_train**4 + X_train**3 + X_train**2 + X_train" >> "Experiment_4.txt"
for i in $(seq 1 $times)
do
    nohup python main.py X_train**4 + X_train**3 + X_train**2 + X_train 4 | grep "best fitness" | tail -1 >> "Experiment_4.txt" 2>&1 &
done
#5

echo "X_train**3 + X_train**2 + X_train" >> "Experiment_5.txt"
for i in $(seq 1 $times)
do
    nohup python main.py X_train**3 + X_train**2 + X_train 5 | grep "best fitness" | tail -1 >> "Experiment_5.txt" 2>&1 &
done
#6

echo "X_train**6 - 2*X_train**4 + X_train**2" >> "Experiment_6.txt"
for i in $(seq 1 $times)
do
    nohup python main.py X_train**6 - 2*X_train**4 + X_train**2 6 | grep "best fitness" | tail -1 >> "Experiment_6.txt" 2>&1 &
done
#7

echo "X_train**5 - 2*X_train**3 + X_train" >> "Experiment_7.txt"
for i in $(seq 1 $times)
do
    nohup python main.py X_train**5 - 2*X_train**3 + X_train 7 | grep "best fitness" | tail -1 >> "Experiment_7.txt" 2>&1 &
done


#Not in function
#1

echo "6.87 + 11*np.cos(7.23*X_train)" >> "Experiment_8.txt"
for i in $(seq 1 $times)
do
    nohup python main.py "6.87 + 11*np.cos(7.23*X_train)" 8 | grep "best fitness" | tail -1 >> "Experiment_8.txt" 2>&1 &
done
#2

echo "np.cos(X_train)*np.sin(X_train**2)-1" >> "Experiment_9.txt"
for i in $(seq 1 $times)
do
    nohup python main.py "np.cos(X_train)*np.sin(X_train**2)-1" 9 | grep "best fitness" | tail -1 >> "Experiment_9.txt" 2>&1 &
done
#3

echo "np.sin(X_train) + np.sin(X_train + X_train**2)" >> "Experiment_10.txt"
for i in $(seq 1 $times)
do
    nohup python main.py "np.sin(X_train) + np.sin(X_train + X_train**2)" 10 | grep "best fitness" | tail -1 >> "Experiment_10.txt" 2>&1 &
done

#TP-, FS-
#1

echo "np.log(X_train)" >> "Experiment_11.txt"
for i in $(seq 1 $times)
do
    nohup python main.py "np.log(np.abs(X_train))" 11 | grep "best fitness" | tail -1 >> "Experiment_11.txt" 2>&1 &
done
#2
echo "np.sqrt(X_train)" >> "Experiment_12.txt"
for i in $(seq 1 $times)
do
    nohup python main.py "np.sqrt(np.abs(X_train))" 12 | grep "best fitness" | tail -1 >> "Experiment_12.txt" 2>&1 &
done

