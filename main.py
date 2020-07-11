"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt
import numpy as np

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3002
MQTT_KEEPALIVE_INTERVAL = 60
DEBUG = True# False


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client(transport="websockets")
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def preprocess(frame, input_shape):
    width, height = input_shape
    output = np.copy(frame)
    output = cv2.resize(output, (width, height))
    output = output.transpose((2, 0, 1))
    output = output[np.newaxis, :, :, :]

    return output

# TODO: Use threshold argument
def detect_persons(result, width, height, threshold=0.8):
    boxes = list(filter(lambda conf: conf[2] > threshold, result[0][0]))
    boxes = sorted(boxes, key=lambda x: x[2])

    persons = []
    for box in boxes:
        image_id, label, conf, x_min, y_min, x_max, y_max = box
        pt1 = (int(x_min * width), int(y_min * height))
        pt2 = (int(x_max * width), int(y_max * height))
        persons.append({ 'pt1': pt1, 'pt2': pt2 })

    return persons

def draw_bounding_boxes(frame, persons):
    for person in persons:
        pt1 = person['pt1']
        pt2 = person['pt2']
        frame = cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 3)

    return frame

def show_inference_time(frame, time, frame_height):
    performance = round(time * 1000, 2);
    text = "Inference: {time} ms".format(time=performance)
    frame = cv2.putText(img=frame,
                        text=text,
                        org=(30, frame_height - 30), # bottom left
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, #COMPLEX_SMALL,
                        fontScale=0.5,
                        color=(139, 0, 0),
                        thickness=1);
    return frame

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device)
    _, _, height, width = infer_network.get_input_shape()

    capture = cv2.VideoCapture(args.input)
    capture.open(args.input)
    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if DEBUG:
        output = cv2.VideoWriter('output.mp4',
                             cv2.VideoWriter_fourcc('M','J','P','G'),
                             24, # fps
                             frameSize=(width, height))


    while capture.isOpened():

        rc, frame = capture.read()

        if not rc:
            break

        preprocessed_frame = preprocess(frame, (width, height))
        start_time = time.time()
        infer_network.exec_net(preprocessed_frame)
        end_time = time.time()

        status = infer_network.wait()

        if status == 0:
            frame = show_inference_time(frame, end_time - start_time, frame_height)
            result = infer_network.get_output()
            persons = detect_persons(result, frame_width, frame_height)
            frame = draw_bounding_boxes(frame, persons)

            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        if DEBUG:
            output.write(frame)

        ### TODO: Write an output image if `single_image_mode` ###

    if DEBUG:
        output.release()

    client.disconnect()
    capture.release()
    cv2.destroyAllWindows()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
