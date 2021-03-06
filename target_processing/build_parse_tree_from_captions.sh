#!/bin/bash
#parameters required: 
#1) file with captions: format expected is nameOfFileToProduce : captionToParse
#2)directory in wich store the generated trees

command="java -mx150m -cp ./*: edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat xmlTree edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz "
direc=$2

#read file and put it into my array
my_array=()
j=0;

input=$1

while IFS="\n" read -r line; do
  my_array+=("$line")
done < $input

len=${#my_array[@]}

#iterate trough string
current_img=""
for (( i=0; i<=$len; i++))
do
	str=${my_array[i]}

	#separate string
	splitted=()

	 while IFS=' : ' read -ra ADDR; do
		for j in "${ADDR[@]}"; do
			#extract string length
			splitted+=($j)
		done

	img_name=${splitted[0]}
	current_img=$img_name
	echo "${splitted[@]:1}" > tmp.txt
	$command tmp.txt > $direc/$img_name

	echo $i

	done <<< "$str"

done
