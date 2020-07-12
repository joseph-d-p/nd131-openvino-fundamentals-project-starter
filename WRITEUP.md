# Project Write-Up

## Custom Layers

The process behind converting custom layers involves extracting each custom layer of the input model.
The extraction process gathers information such as input, output and parameters for each layer.
The model is then optimized given the extracted information.
The generated optimized model is the input for the Inference Engine.

Some of the potential reasons for handling custom layers are:
1. Be able to generate a better optimized model
2. Be flexible in handling different input model formats and structure


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:
* Count number of people in bus stops - the number of people and the average duration may have some relationship with bus arrival times and vehicle traffic.
* Count number of people inside a restaurant - the count explains how busy the restaurant is and table turnover time.
* Count number of patients in waiting area - the count can help give feedback to patients planning to go for an appointment.


## Assess Effects on End User Needs

* I would recommend this system to be used indoors as the lighting environment can be managed unlike outdoors where environment conditions affect lighting.
* Image size and frame rate affects the accuracy and speed of inference.
* The camera orientation can be better configured indoors unlike outdoors like there can be random object obstructions that can skew analysis of the count and duration.

## Model Research

All models used are from the [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Faster RCNN Inception V2](https://github.com/joseph-d-p/people-counter-app/tree/master/models/faster_rcnn_inception_v2_coco)
  - [Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
  ```
  /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
    --input_model frozen_inference_graph.pb \
    --model_name faster_rcnn_inception_v2 \
    --reverse_input_channels \
    --tensorflow_object_detection_api_pipeline_config pipeline.config \
    --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
  ```
  - The model was insufficient for the app because it had a segmentation fault when running the inference request. The model had two inputs: `image_info` and `image_tensor` of shapes [1,3] and [1,3,600,600] respectively.
  
- Model 2: [SSD MobileNet V2](https://github.com/joseph-d-p/people-counter-app/tree/master/models/ssd_mobilenet_v2_coco)
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
  ```
  /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
    --input_model frozen_inference_graph.pb \
    --model_name ssd_mobilenet_v2 \
    --reverse_input_channels \
    --tensorflow_object_detection_api_pipeline_config pipeline.config \
    --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model was insufficient for the app because of its low precision.

- Model 3: [Mask RCNN Inception](https://github.com/joseph-d-p/people-counter-app/tree/master/models/mask_rcnn_inception_v2_coco)
  - [Model Source](http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
  ```
  /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py \
    --input_model frozen_inference_graph.pb \
    --model_name mask_rcnn_inception_v2 \
    --reverse_input_channels \
    --tensorflow_object_detection_api_pipeline_config pipeline.config \
    --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/mask_rcnn_support.json
  ```
  - The model was insufficient for the app because it had a segmentation fault when running the inference request. The model had two inputs: `image_info` and `image_tensor` of shapes [1,3] and [1,3,800,800] respectively.

## Comparing Performance

Using Intel's [person-detection-retail-0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html) detection vs SSD Mobilenet V2.

**person-detection-retail-0013**:
- input shape: [1, 3, 320, 544]
- speed: 1 ms

Command used:
```
python main.py \
  -i resources/Pedestrian_Detect_2_1_1.mp4 \
  -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml \
  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib \
  -d CPU \
  -pt 0.7 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

**ssd_mobilenet_v2**:
- input shape: [1, 3, 300, 300]
- speed before optimization: 31 ms (based on TF's performance table)
- speed after optimization: < 1 ms

Command used:
```
python main.py \
  -i resources/Pedestrian_Detect_2_1_1.mp4 \
  -m models/ssd_mobilenet_v2_coco/ssd_mobilenet_v2.xml \
  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib \
  -d CPU \
  -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Conclusion

I ended up using Intel's model due to the accuracy in detecting people and it's still relatively fast.
