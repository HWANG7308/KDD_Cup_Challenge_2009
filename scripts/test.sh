# Author: S1802373

# test model
if [ $# -lt 1 ]; then
	echo "Usage: $0 [one of the three prediction tasks]"
    echo "a for appetency; c for churn; u for upselling"
	echo "e.g. bash test.sh a (for appetency)"
    exit 1
fi

file_select=$1

#python ../code/code.py $file_select
if [ $file_select == "a" ]; then
    echo "----- Appetency -----"
    python ../code/main.py $file_select
elif [ $file_select == "c" ]; then
    echo "----- Churn -----"
    python ../code/main.py $file_select
elif [ $file_select == "u" ]; then
    echo "----- Upselling -----"
    python ../code/main.py $file_select
else
    echo "Wrong"
fi
