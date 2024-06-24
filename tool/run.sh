#! /bin/sh


for py_file in $(find ./ga_hls/$1 -name '*.py')
do 
	echo $py_file
	echo python3 ga_hls $1 $py_file $2 ./ga_hls/$1/$3.json

    python3 ga_hls $1 $py_file $2 ./ga_hls/$1/$3.json
done
