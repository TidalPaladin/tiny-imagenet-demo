#!/bin/sh
lines=$(cat "class_list.txt" | awk '{print $1}')
place=4
unique_lines=$(echo "$lines" | grep -oE "n[0-9]{$place}" | uniq -u | wc -l)
echo "Unique classes = $unique_lines"

