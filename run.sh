#! /bin/sh

for py_file in $(find ./ga_hls/benchmark -name *.py)
do 
	echo $py_file
    python3 $py_file
done