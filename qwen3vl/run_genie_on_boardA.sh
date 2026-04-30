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

for i in $(seq 14 14); do
    INPUT_FILE="inputs_embeds_${i}_uint16.bin"
    RESULT_JSON="result_${i}.json"
    RESULT_LOG="genie_${i}.log"

    echo "[INFO] Running inference for $INPUT_FILE"
    NETRUN_COMMAND="${DEPLOY_DIR}/bin/genie-t2t-run \
        -c qwen3vl.json \
        -e test_data_onboard_1664/${INPUT_FILE},uint16,${SCALE},${OFFSET} \
        --log error \
        --profile ${RESULT_JSON}"

    # Clear logcat and start capture
    # logcat -c
    # logcat | grep -i genie > "logcat.log" 2>&1 &

    # Run command and log output
    echo "$NETRUN_COMMAND" >> "${RESULT_LOG}" 2>&1
    $NETRUN_COMMAND >> "${RESULT_LOG}" 2>&1
    RETURN_VAL=$?

    if [ $RETURN_VAL -eq 0 ]; then
        echo "Run ${i} successfully for ${INPUT_FILE}" >> "${RESULT_LOG}"
    else
        echo "Run ${i} failed for ${INPUT_FILE}" >> "${RESULT_LOG}"
    fi

    # Optional: extract performance metrics
    echo "Performance result for ${INPUT_FILE}:"
    grep -E -A 3 "num-prompt-tokens|prompt-processing-rate|time-to-first-token|token-generation-rate|token-acceptance-rate" "${RESULT_JSON}"

    # Kill background logcat process
    
    journalctl -b --since "$START_TIME" | grep -i genie > journalctl.log 2>&1


    sleep 1
done

echo "[DONE] All runs completed."