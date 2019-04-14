
#!/bin/bash

echo ""
echo "------------------------------------------------"
echo "     [Install - Train - Clean] - Script "
echo "------------------------------------------------"
echo ""
echo ""


if [ $1 = "install" ]; then
	#-------------------------------------------------------------------------------------------
	# Install Python Version
	#-------------------------------------------------------------------------------------------

	sudo apt install python

	#-------------------------------------------------------------------------------------------
	# Install All Modules For Collect
	#-------------------------------------------------------------------------------------------

	sudo apt-get install python-pip
	pip install opencv-python
	pip install pillow
	pip install psutil

	#-------------------------------------------------------------------------------------------
	# Install Tensorflow
	#-------------------------------------------------------------------------------------------

	sudo apt-get install python-pip python-dev python-virtualenv 
	sudo virtualenv --system-site-packages ~/tensorflow
	sudo pip install --upgrade tensorflow

	#-------------------------------------------------------------------------------------------
	# Get Trainer
	#-------------------------------------------------------------------------------------------

	sudo apt install curl
	curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py

	#-------------------------------------------------------------------------------------------
	# Test Modules
	#-------------------------------------------------------------------------------------------

	python verify.py

	#-------------------------------------------------------------------------------------------
	# Move To Next Stage - Collect Data With collect.py
	#-------------------------------------------------------------------------------------------

	echo ""
	echo ""





elif [ $1 = "train" ]; then
	#-------------------------------------------------------------------------------------------
	# Train
	#-------------------------------------------------------------------------------------------

	python retrain.py \
	  --bottleneck_dir=bottlenecks \
	  --how_many_training_steps=500 \
	  --model_dir=inception \
	  --summaries_dir=training_summaries/basic \
	  --output_graph=retrained_graph.pb \
	  --output_labels=retrained_labels.txt \
	  --image_dir=images

	mv inception/ data/
	mv bottlenecks/ data/
	mv training_summaries/ data/
	mv retrained_graph.pb data/
	mv retrained_labels.txt data/


	#-------------------------------------------------------------------------------------------
	# Bottlenecks Created - Run Identifier
	#-------------------------------------------------------------------------------------------

	echo ""
	echo "ALL IMAGES TRAINED: BOTTLENECKS CREATED"




elif [ $1 = "clean" ]; then
	#-------------------------------------------------------------------------------------------
	# Cleanup Previous Train
	#-------------------------------------------------------------------------------------------

	rm -r images
	mkdir images

	rm -r target
	mkdir target

	cd target
	rm -r display
	mkdir display
	cd ..

	rm -r rejects
	mkdir rejects
	
	cd data
	rm -r training_summaries
	rm -r inception
	rm -r bottlenecks

	rm retrained_graph.pb
	rm retrained_labels.txt
	#-------------------------------------------------------------------------------------------
	# Default State Restored
	#-------------------------------------------------------------------------------------------

	echo ""
	echo "PROGRAM FILES CLEANED"



else
	echo "NOT A VALID COMMAND"

fi
