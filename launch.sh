#! /bin/sh

listVar="AT1 AT2 AT51 AT52 AT53 AT54 AT6A AT6B AT6C CC1 CC2 CC3 CC5 CCX"

exp='exp1'


for i in $listVar; do
    echo "$i"
    sh run.sh $i $1 $exp
done