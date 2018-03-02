#   Copyright (c) 2016, Xilinx, Inc.
#   All rights reserved.
# 
#   Redistribution and use in source and binary forms, with or without 
#   modification, are permitted provided that the following conditions are met:
#
#   1.  Redistributions of source code must retain the above copyright notice, 
#       this list of conditions and the following disclaimer.
#
#   2.  Redistributions in binary form must reproduce the above copyright 
#       notice, this list of conditions and the following disclaimer in the 
#       documentation and/or other materials provided with the distribution.
#
#   3.  Neither the name of the copyright holder nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#   THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pynq import Overlay, PL
from PIL import Image
from collections import OrderedDict
from bnn.dataset.mnist import load_mnist
import numpy as np
import cffi
import os
import tempfile
import time

BNN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
BNN_LIB_DIR = os.path.join(BNN_ROOT_DIR, 'libraries')
BNN_BIT_DIR = os.path.join(BNN_ROOT_DIR, 'bitstreams')
BNN_PARAM_DIR = os.path.join(BNN_ROOT_DIR, 'params')

RUNTIME_HW = "python_hw"
RUNTIME_SW = "python_sw"
NETWORK_CNV = "cnv-pynq"
NETWORK_LFC = "lfc-pynq"
NETWORK_ADD = "add-pynq"
NETWORK_ADD_DOUBLE = "add-double-pynq"
NETWORK_FC = "fc-pynq"

FC_INPUT_SIZE = 784;
FC_HIDDEN1_SIZE = 25;
FC_OUTPUT_SIZE = 10;
FC_BATCH_SIZE = 40;
FC_TRAIN_SIZE = 60000;
FC_TEST_SIZE = 10000;
FC_W1_SIZE = FC_INPUT_SIZE * FC_HIDDEN1_SIZE;
FC_B1_SIZE = FC_HIDDEN1_SIZE;
FC_W2_SIZE = FC_HIDDEN1_SIZE * FC_OUTPUT_SIZE;
FC_B2_SIZE = FC_OUTPUT_SIZE;
FC_W_B_SIZE = FC_W1_SIZE + FC_B1_SIZE + FC_W2_SIZE + FC_B2_SIZE;


_ffi = cffi.FFI()

_ffi.cdef("""
void load_parameters(const char* path);
unsigned int inference(const char* path, unsigned int results[64], int number_class, float *usecPerImage);
unsigned int* inference_multiple(const char* path, int number_class, int *image_number, float *usecPerImage, unsigned int enable_detail);
void free_results(unsigned int * result);
void deinit();
"""
)

_ffi_add = cffi.FFI()
_ffi_add.cdef("""
unsigned int *add(unsigned int in1, unsigned int in2, float *usecPerMul);
void free_results(unsigned int *result);
void deinit();
"""
)

_ffi_add_double = cffi.FFI()
_ffi_add_double.cdef("""
double *add_double(double in1, double in2, float *usecPerMul);
void free_results(double *result);
void deinit();
"""
)

_ffi_fc = cffi.FFI()
_ffi_fc.cdef("""
void load_images(const char* path);
float *train(unsigned int imageNum, float *usecPerMul);
void free_results(float *result);
void free_images();
void deinit();
"""
)

_libraries = {}

class PynqBNN:
    
    def __init__(self, runtime=RUNTIME_HW, network=NETWORK_CNV, load_overlay=True):
        self.bitstream_name = None
        if runtime == RUNTIME_HW:
            self.bitstream_name="{0}-pynq.bit".format(network)
            self.bitstream_path=os.path.join(BNN_BIT_DIR, self.bitstream_name)
            if PL.bitfile_name != self.bitstream_path:
                if load_overlay:
                    Overlay(self.bitstream_path).download()
                else:
                    raise RuntimeError("Incorrect Overlay loaded")
        dllname = "{0}-{1}.so".format(runtime, network)
        if dllname not in _libraries:
            if network == NETWORK_ADD:
                _libraries[dllname] = _ffi_add.dlopen(
                    os.path.join(BNN_LIB_DIR, dllname))
            elif network == NETWORK_ADD_DOUBLE:
                _libraries[dllname] = _ffi_add_double.dlopen(
                    os.path.join(BNN_LIB_DIR, dllname))
            elif network == NETWORK_FC:
                _libraries[dllname] = _ffi_fc.dlopen(
                    os.path.join(BNN_LIB_DIR, dllname))
            else:
                _libraries[dllname] = _ffi.dlopen(
		    os.path.join(BNN_LIB_DIR, dllname))
        self.interface = _libraries[dllname]
        self.num_classes = 0
        
    def __del__(self):
        self.interface.deinit()
        
    def load_parameters(self, params):
        if not os.path.isabs(params):
            params = os.path.join(BNN_PARAM_DIR, params)
        self.interface.load_parameters(params.encode())
        self.classes = []
        with open (os.path.join(params, "classes.txt")) as f:
            self.classes = [c.strip() for c in f.readlines()]
        filter(None, self.classes)
        
    def inference(self, path):
        usecperimage = _ffi.new("float *") 
        result_ptr = self.interface.inference(path.encode(), _ffi.NULL, len(self.classes), usecperimage)
        print("Inference took %.2f microseconds" % (usecperimage[0]))
        print("Classification rate: %.2f images per second" % (1000000.0/usecperimage[0]))
        return result_ptr

    def detailed_inference(self, path):
        details_ptr = _ffi.new("unsigned int[]", len(self.classes))
        usecperimage = _ffi.new("float *") 
        self.interface.inference(path.encode(), details_ptr, len(self.classes), usecperimage)
        details_buf = _ffi.buffer(details_ptr, len(self.classes) * 4)
        print("Inference took %.2f microseconds" % (usecperimage[0]))
        print("Classification rate: %.2f images per second" % (1000000.0/usecperimage[0]))
        details_array = np.copy(np.frombuffer(details_buf, dtype=np.uint32))
        return details_array

    def inference_multiple(self, path):
        size_ptr = _ffi.new("int *")
        usecperimage = _ffi.new("float *")
        result_ptr = self.interface.inference_multiple(
            path.encode(), len(self.classes), size_ptr, usecperimage,0)
        result_buffer = _ffi.buffer(result_ptr, size_ptr[0] * 4)
        print("Inference took %.2f microseconds, %.2f usec per image" % (usecperimage[0]*size_ptr[0],usecperimage[0]))
        result_array = np.copy(np.frombuffer(result_buffer, dtype=np.uint32))
        print("Classification rate: %.2f images per second" % (1000000.0/usecperimage[0]))
        self.interface.free_results(result_ptr)
        return result_array
    
    def inference_multiple_detail(self, path):
        size_ptr = _ffi.new("int *")
        usecperimage = _ffi.new("float *")
        result_ptr = self.interface.inference_multiple(
            path.encode(), len(self.classes), size_ptr, usecperimage,1)
        
        print("Inference took %.2f microseconds, %.2f usec per image" % (usecperimage[0]*size_ptr[0],usecperimage[0]))
        print("Classification rate: %.2f images per second" % (1000000.0/usecperimage[0]))
        result_buffer = _ffi.buffer(result_ptr,len(self.classes)* size_ptr[0] * 4)
        result_array = np.copy(np.frombuffer(result_buffer, dtype=np.uint32))
        self.interface.free_results(result_ptr)
        return result_array

    def class_name(self, index):
        return self.classes[index]

    def add(self, in1, in2):
        usecpermult = _ffi_add.new("float *")
        result_ptr = self.interface.add(in1, in2, usecpermult)
        print("bnn.add(): Addition took %.2f microseconds" % (usecpermult[0]))
        result_buffer = _ffi_add.buffer(result_ptr)
        result_array = np.copy(np.frombuffer(result_buffer, dtype=np.uint32))
        self.interface.free_results(result_ptr)
        return result_array

    def add_double(self, in1, in2):
        usecpermult = _ffi_add_double.new("float *")
        result_ptr = self.interface.add_double(in1, in2, usecpermult)
        print("bnn.add_double(): Addition took %.2f microseconds" % (usecpermult[0]))
        result_buffer = _ffi_add_double.buffer(result_ptr)
        result_array = np.copy(np.frombuffer(result_buffer, dtype=np.float64))
        self.interface.free_results(result_ptr)
        return result_array

    def train(self, path="/home/xilinx/wtakase/mnist",
              image_num=FC_BATCH_SIZE, epoch_num=None,
              get_accuracy=True):
        self.interface.load_images(path.encode())
        usecpermult = _ffi_fc.new("float *")
        if epoch_num is None:
            if image_num > FC_TRAIN_SIZE:
                image_num = FC_TRAIN_SIZE
            loop_per_image_num = FC_BATCH_SIZE
            loop_num = int(image_num / FC_BATCH_SIZE)
            if image_num % FC_BATCH_SIZE != 0:
                loop_num += 1
        else:
            loop_per_image_num = FC_TRAIN_SIZE
            loop_num = epoch_num
            if loop_num <= 0:
                loop_num = 1

        result_arrays = []
        total_start_time = time.time()
        for i in range(loop_num):
            sub_start_time = time.time()
            result_ptr = self.interface.train(loop_per_image_num, usecpermult)
            sub_end_time = time.time()
            #print(" %d-images training took %.2f sec" % (loop_per_image_num,
            #                                            sub_end_time - sub_start_time))
            result_buffer = _ffi_fc.buffer(result_ptr, FC_W_B_SIZE * 4)
            result_array = np.copy(np.frombuffer(result_buffer, dtype=np.float32))
            result_arrays.append({"image_num": (i + 1) * loop_per_image_num,
                                  "result": result_array})
            self.interface.free_results(result_ptr)
        total_end_time = time.time()
        print("%d-images training took %.2f sec" % (loop_num * loop_per_image_num,
                                                    total_end_time - total_start_time))
        self.interface.free_images()

        w_bs = []
        for result in result_arrays:
            w1 = result["result"][0:FC_W1_SIZE]
            b1 = result["result"][FC_W1_SIZE:FC_W1_SIZE+FC_B1_SIZE]
            w2 = result["result"][FC_W1_SIZE+FC_B1_SIZE:FC_W1_SIZE+FC_B1_SIZE+FC_W2_SIZE]
            b2 = result["result"][FC_W1_SIZE+FC_B1_SIZE+FC_W2_SIZE:FC_W1_SIZE+FC_B1_SIZE+FC_W2_SIZE+FC_B2_SIZE]
            w_bs.append({"image_num": result["image_num"],
                         "w1": w1, "b1": b1, "w2": w2, "b2": b2})

        if get_accuracy:
            return self.get_accuracy(w_bs)
        else:
            return w_bs

    def get_accuracy(self, w_bs):
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)
        #iter_num = int(FC_TEST_SIZE / FC_BATCH_SIZE)
        iter_num = 10
        accuracies = []
        for w_b in w_bs:
            network = TwoLayerNet(w_b["w1"], w_b["b1"], w_b["w2"], w_b["b2"])
            accuracy = 0.0
            for i in range(iter_num):
                batch_mask = np.random.choice(FC_TEST_SIZE, FC_BATCH_SIZE)
                x_batch = x_test[batch_mask] / 255.0
                t_batch = t_test[batch_mask]
                accuracy += network.accuracy(x_batch, t_batch)
            accuracies.append({"image_num": w_b["image_num"],
                               "accuracy": accuracy / iter_num})
        return accuracies


class CnvClassifier:
    def __init__(self, params, runtime=RUNTIME_HW):
        self.bnn = PynqBNN(runtime, network=NETWORK_CNV)
        self.bnn.load_parameters(params)
    
    def image_to_cifar(self, im, fp):
        # We resize the downloaded image to be 32x32 pixels as expected from the BNN
        im.thumbnail((32, 32), Image.ANTIALIAS)
        background = Image.new('RGBA', (32, 32), (255, 255, 255, 0))
        background.paste(
            im, (int((32 - im.size[0]) / 2), int((32 - im.size[1]) / 2))
        )
        # We write the image into the format used in the Cifar-10 dataset for code compatibility 
        im = (np.array(background))
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        label = np.identity(1, dtype=np.uint8)
        fp.write(label.tobytes())
        fp.write(r.tobytes())
        fp.write(g.tobytes())
        fp.write(b.tobytes())
    
    def classify_image(self, im):
        with tempfile.NamedTemporaryFile() as tmp:
            self.image_to_cifar(im, tmp)
            tmp.flush()
            return self.bnn.inference(tmp.name)
    
    def classify_details(self, im):
        with tempfile.NamedTemporaryFile() as tmp:
            self.image_to_cifar(im, tmp)
            tmp.flush()
            return self.bnn.detailed_inference(tmp.name)

    def classify_path(self, path):
        im = Image.open(path)
        return self.classify_image(im)
    
    def classify_images(self, ims):
        with tempfile.NamedTemporaryFile() as tmp:
            for im in ims:
                self.image_to_cifar(im, tmp)
            tmp.flush()
            return self.bnn.inference_multiple(tmp.name)
    
    def classify_images_details(self, ims):
        with tempfile.NamedTemporaryFile() as tmp:
            for im in ims:
                self.image_to_cifar(im, tmp)
            tmp.flush()
            return self.bnn.inference_multiple_detail(tmp.name)
    
    def classify_paths(self, paths):
        return self.classify_images([Image.open(p) for p in paths])
    
    def class_name(self, index):
        return self.bnn.classes[index]

def available_params(network):
    options = os.listdir(BNN_PARAM_DIR)
    ret = []
    for d in options:
        if os.path.isdir(os.path.join(BNN_PARAM_DIR, d)):
            test_lfc = os.path.join(BNN_PARAM_DIR, d, '1-63-thres.bin')
            test_cnv = os.path.join(BNN_PARAM_DIR, d, '8-3-thres.bin')
            if network == NETWORK_LFC and os.path.exists(test_lfc):
               ret.append(d)
            if network == NETWORK_CNV and os.path.exists(test_cnv):
               ret.append(d)
    return ret


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        # If x = [1, 2, 0, -1], (x <= 0) returns [False, False, True, True].
        self.mask = (x <= 0)
        # Necessary to avoid overriding original x values.
        out = x.copy()
        # Set 0, if out[i] == True
        out[self.mask] = 0

        return out


class TwoLayerNet:
    def __init__(self, w1, b1, w2, b2,
                 input_size=FC_INPUT_SIZE,
                 hidden_size=FC_HIDDEN1_SIZE,
                 output_size=FC_OUTPUT_SIZE):
        self.params = {}
        self.params['W1'] = w1.reshape((input_size,hidden_size))
        self.params['b1'] = b1
        self.params['W2'] = w2.reshape((hidden_size, output_size))
        self.params['b2'] = b2

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        return  np.sum(y == t) / float(x.shape[0])
