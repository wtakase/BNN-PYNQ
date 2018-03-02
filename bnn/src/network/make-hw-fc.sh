#!/bin/sh

rm -rf output/hls-syn/fc-pynq
rm -rf output/vivado/fc-pynq-pynq
./make-hw.sh fc-pynq pynq a
