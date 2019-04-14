
#!/bin/bash
cd ..

#-------------------------------------------------------------------------------------------
# Cleanup Previous Train
#-------------------------------------------------------------------------------------------

rm -r training_summaries
rm -r inception
rm -r bottlenecks

rm retrained_graph.pb
rm retrained_labels.txt

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


