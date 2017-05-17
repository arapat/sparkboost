./spark/bin/spark-submit --master spark://ec2-54-173-228-241.compute-1.amazonaws.com:7077 \
--class sparkboost.examples.SpliceSite --conf spark.executor.extraJavaOptions=-XX:+UseG1GC \
./sparkboost_2.11-0.1.jar --train /train-txt --test /test-txt --sample-frac 0.1 \
--cores 80 --num-slices 2 --max-iteration 0 --algorithm 1 --save-model ./model.bin \
--save-train-rdd /train2 --save-test-rdd /test2 --data-source 2 \
--improve 0.01 --load-train-rdd /train2 --load-test-rdd /test2


