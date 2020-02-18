#!/bin/bash

bin_path=/glia/bin;
gPb_path=/dataset/msCoco/gPb/
gray_images=/dataset/msCoco/gray/
new_path=/dataset/msCoco/pipeline/;
model=/glia/code/models/level_0.15_blur_6;
final_results=/dataset/msCoco/finals/;

file_names=()
#extract file_names
while IFS='\n' read -r line || [[ -n "$line" ]]; do
  file_names+=("$line")
done < "$1"

dataSet_len=${#file_names[@]}

#pipeline
i=0
for f_name in ${file_names[@]}
do

	len=${#f_name}

	name=$f_name

	echo $i  $name
	
	#data

	level=0.15

	pbb=$gPb_path$name"_blurred_6.png";

	pb=$gPb_path$name"_gPb.png";

	segii=$new_path$name"_initial_seg_blurred.mha";

	segi=$new_path$name"_initial_seg_postMerged.mha";
				
	order=$new_path$name"_order.txt"

	sal=$new_path$name"_sal.txt" 

	bcf=$new_path$name"_boundary_features.txt"

	gray=$gray_images$name".jpg"

	bcp=$new_path$name"_predictions.txt"

	ris=$final_results$name"_"$"_segmentation"

	#pipeline


	$bin_path"/watershed" -i $pbb -l $level -o $segii

	$bin_path"/pre_merge" -s $segii -p $pb -t 50 200 -b 0.5 -r true -o $segi

	$bin_path"/merge_order_pb" -s $segi -p $pb -o $order -y $sal

	$bin_path"/bc_feat" -s $segi -o $order -y $sal --pb $pb -b $bcf --bt 0.2 0.5 0.8 --rbi $pb --rbb 16 --rbl 0.0 --rbu 1.0  --rbi $gray --rbb 16 --rbl 0.0 --rbu 1.0

	$bin_path"/pred_rf" --m $model --l -1 --f $bcf --p $bcp

	$bin_path"/segment_ccm" -s $segi -o $order -p $bcp -r true  -f $ris

	segments=()
	while IFS='\n' read -r line || [[ -n "$line" ]]; do
	  segments+=("$line")
	done < $ris.sif

	n_segments=${#segments[@]}
	echo segment found $n_segments
	if [ $n_segments -lt 2  ]
	  then
	    rm $ris.mha
	    rm $ris.sif
	    echo deleted!
	fi

	#rm $new_path/*

	((i++))

done
