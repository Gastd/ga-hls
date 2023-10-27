#! /bin/sh

for py_file in $(find ./ga_hls/$1 -name '*.py')
do 
	echo $py_file
    python3 ga_hls $1 $py_file
done
