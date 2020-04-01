#!/bin/bash

data_set_part=( "val/" "test/" "train/" )

#insert your path
bin_path=	#dir containing glia executables
gPb_path=	#dir containing gPb computed on original images
gray_images=	#dir containing gray scaled images of the original images
new_path=	#dir that will contain the temporary results of the pipeline (its content could be deleted after the process termination)
dataSet_path=	#dir containing the dataset
truths=		#dir containing the ground truth segmentations

for part in 2; do

	for level in 0.05 0.1 0.15; do

		for blur in 5; do

			current_part=${data_set_part[part]};

			mkdir $new_path$current_part"level_"$level"_blur_"$blur
			
			current_path=$new_path$current_part"level_"$level"_blur_"$blur/;

			echo $current_path
		
	 		for i in {0..199}; do 

				echo $i
	
				#extract name

				file=$(ls -1 $gPb_path$current_part | grep "el_"$i"_" | grep "_gPb.png")

				num_len=${#i}
				
				name=${file:4+$num_len:-8} 

				echo $name

				#data

				segii=$current_path"el_"$i"_"$name"_initial_seg_blurred.mha";

				pbb=$gPb_path$current_part"el_"$i"_"$name"_blurred_"$blur".png";

				pb=$gPb_path$current_part"el_"$i"_"$name"_gPb.png";

				segi=$current_path"el_"$i"_"$name"_initial_seg_postMerged.mha";

				order=$current_path"el_"$i"_"$name"_order.txt"

				sal=$current_path"el_"$i"_"$name"_sal.txt" 

				bcf=$current_path"el_"$i"_"$name"_boundary_features.txt"

				gray=$gray_images$current_part$name".jpg"

				#invoke each step

				$bin_path"/watershed" -i $pbb -l $level -o $segii 

				$bin_path"/pre_merge" -s $segii -p $pb -t 50 200 -b 0.5 -r true -o $segi

				$bin_path"/merge_order_pb" -s $segi -p $pb -o $order -y $sal

				$bin_path"/bc_feat" -s $segi -o $order -y $sal --pb $pb -b $bcf --bt 0.2 0.5 0.8 & --rbi $pb --rbb 16 --rbl 0.0 --rbu 1.0 --rbi $gray --rbb 16 --rbl 0.0 --rbu 1.0

				n_truth=$(ls -1 $truths$current_part | grep "im"$name | wc -l);		

				#for all ground truth extract labels

				for (( j=1; j<=$n_truth/2; j++))
				do 
					truth=$truths$current_part"im"$name"_"$j"_segmentation.png"

					label=$current_path"el_"$i"_"$name"_boundary_labels_"$j".txt";

					$bin_path"/bc_label_ri" -s $segi -o $order -t $truth -l $label &
			
				done
			done
		done	
	done
done


