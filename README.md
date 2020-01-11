# image_captionig_tree2tree_input-target_processing
Code for input and output processing of  image_captionig_tree2tree https://github.com/dave94-42/image_captionig_tree2tree

More informations about how to use the code:

gPb algorithm on original images: to obtain gPb of original images (need by glia pipeline) I have used https://github.com/vrabaud/gPb but you can also use the official matlab implementation of this algorithm

input_processing/glia_pipeline: 
        
	Download Graph Learning Library for Image Analysis (GLIA) from https://github.com/tingliu/glia in to input_processing/glia_pipeline directory and its dependency:
	
	In input_processing/glia_pipeline/glia_pipeline.sh is needed to specify gPb_path,gray_images, new_path, model,final_results that are respectively directory of gPb algorithm applied to the original images, original images in gray scale, path in which you store the intermediate result of the pipeline (could be removed once you have final results), file containing the model for random forest, and final result (i.e. file describing the segmentation tree and segmentation map file)
	
input_processing/input_tree_label: after previous stage you get in the final result directory what is needed, along with the original images, to label crated trees with convolutional informations coming form alexNet. More info by running the code with --help argument
