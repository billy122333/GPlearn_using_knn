#!/bin/bash

# 清空之前的輸出文件，如果需要
OUTPUT_PATH="output/output20_30times_PathSampling.txt"
times=2

> "$OUTPUT_PATH" 


# In function
#1
echo "1.57+2.43*X_train" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py "1.57+2.43*X_train"  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#2
echo "X_train**6 + X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py X_train**6 + X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#3
echo "X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py X_train**5 + X_train**4 + X_train**3 + X_train**2 + X_train  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#4
echo "X_train**4 + X_train**3 + X_train**2 + X_train" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py X_train**4 + X_train**3 + X_train**2 + X_train  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#5
echo "X_train**3 + X_train**2 + X_train" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py X_train**3 + X_train**2 + X_train  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#6
echo "X_train**6 - 2*X_train**4 + X_train**2" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py X_train**6 - 2*X_train**4 + X_train**2  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#7
echo "X_train**5 - 2*X_train**3 + X_train" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py X_train**5 - 2*X_train**3 + X_train  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done


#Not in function
#1
echo "6.87 + 11*np.cos(7.23*X_train)" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py "6.87 + 11*np.cos(7.23*X_train)"  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#2
echo "np.cos(X_train)*np.sin(X_train**2)-1" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py "np.cos(X_train)*np.sin(X_train**2)-1"  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#3
echo "np.sin(X_train) + np.sin(X_train + X_train**2)" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py "np.sin(X_train) + np.sin(X_train + X_train**2)"  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done

#TP-, FS-
#1
echo "np.log(X_train)" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py "np.log(np.abs(X_train))"  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done
#2
echo "np.sqrt(X_train)" >> "$OUTPUT_PATH"
for i in $(seq 1 $times)
do
    python main.py "np.sqrt(np.abs(X_train))"  | grep "best fitness" | tail -1 >> "$OUTPUT_PATH"
done

python test_scripts/count_mean.py "$OUTPUT_PATH" >> "$OUTPUT_PATH"