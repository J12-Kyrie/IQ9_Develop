#!/bin/sh
# This script should be executed on board

if [ ! "$DEPLOY_DIR" ]; then
    DEPLOY_DIR=$(pwd)
fi

echo "DEPLOY_DIR=$DEPLOY_DIR"

cd "$DEPLOY_DIR" || exit
cd "$DEPLOY_DIR" || exit
export LD_LIBRARY_PATH="$DEPLOY_DIR/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$DEPLOY_DIR/lib/dsp;/vendor/dsp/cdsp;/vendor/dsp/cdsp0"
export CDSP_LIBRARY_PATH="$DEPLOY_DIR/lib/dsp;/vendor/dsp/cdsp;/vendor/dsp/cdsp0"
export CDSP1_LIBRARY_PATH="$DEPLOY_DIR/lib/dsp;/vendor/dsp/cdsp1"
export CDSP2_LIBRARY_PATH="$DEPLOY_DIR/lib/dsp;/vendor/dsp/cdsp2"
export CDSP3_LIBRARY_PATH="$DEPLOY_DIR/lib/dsp;/vendor/dsp/cdsp3"

# Known quantization parameters
# SCALE="2.5988021661760285e-05"
# OFFSET="-21043"
SCALE="0.00037788687041029334"
OFFSET="-36275"


# Clean old logs
rm -rf gen*.log result*.json
sleep 1

echo "Starting batch run for inputs_embeds_0_uint16.bin to inputs_embeds_9_uint16.bin"

INPUT_FILE="inputs_embeds_14_uint16.bin"
RESULT_JSON="result_14.json"
RESULT_LOG="genie_14.log"

${DEPLOY_DIR}/bin/genie-t2t-run \
        -c qwen3vl.json \
        -e test_data_onboard_1664/${INPUT_FILE},uint16,${SCALE},${OFFSET} \
        --log verbose 


