#!/bin/sh
#
# Usage: ./rebuild-hw.sh cnv-pynq pynq
#
# Before running this script, you need to run the following command:
# $ ./make-hw.sh cnv-pynq pynq a
# Then, open the project, modify, and save it.
# $ vivado output/vivado/cnv-pynq-pynq/cnv-pynq-pynq.xpr

NETWORK=$1
PLATFORM=$2

if [ -z "$XILINX_BNN_ROOT" ]; then
    export XILINX_BNN_ROOT="$( ( cd "$(dirname "$0")/.."; pwd) )"
fi

BNN_PATH=$XILINX_BNN_ROOT/network

VIVADO_SCRIPT_DIR=$XILINX_BNN_ROOT/library/script/
VIVADO_SCRIPT=$VIVADO_SCRIPT_DIR/rebuild-vivado-proj.tcl

TARGET_NAME="$NETWORK-$PLATFORM"
VIVADO_OUT_DIR="$BNN_PATH/output/vivado/$TARGET_NAME"
VIVADO_PROJECT="$VIVADO_OUT_DIR/$NETWORK-$PLATFORM.xpr"
BITSTREAM_PATH="$BNN_PATH/output/bitstream"
TARGET_BITSTREAM="$BITSTREAM_PATH/$NETWORK-$PLATFORM.bit"
TARGET_TCL="$BITSTREAM_PATH/$NETWORK-$PLATFORM.tcl"

vivado -mode batch -source $VIVADO_SCRIPT -tclargs $VIVADO_PROJECT

cp -f "$VIVADO_OUT_DIR/$TARGET_NAME.runs/impl_1/procsys_wrapper.bit" $TARGET_BITSTREAM
cp -f "$VIVADO_OUT_DIR/procsys.tcl" $TARGET_TCL
echo "Bitstream copied to $TARGET_BITSTREAM"

echo "Done!"

exit 0
