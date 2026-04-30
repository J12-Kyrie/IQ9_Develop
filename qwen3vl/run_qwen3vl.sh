#!/bin/bash
# Qwen3-VL on QCS9075 - end-to-end inference script
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$SCRIPT_DIR/lib/dsp:/usr/lib/dsp/cdsp:/usr/lib/dsp/cdsp1"
SCALE="0.00037788687041029334"
OFFSET="-36275"
IDX="${1:-14}"
INPUT_FILE="test_data_onboard_1664/inputs_embeds_${IDX}_uint16.bin"
RESULT_JSON="result_${IDX}.json"
echo "[Qwen3-VL] Running inference on embedding #${IDX}..."
sg fastrpc -c "cd $SCRIPT_DIR && timeout 300 ${SCRIPT_DIR}/bin/genie-t2t-run -c ${SCRIPT_DIR}/qwen3vl.json -e ${SCRIPT_DIR}/${INPUT_FILE},uint16,${SCALE},${OFFSET} --log error --profile ${SCRIPT_DIR}/${RESULT_JSON}"
echo "[Qwen3-VL] Done. Profile saved to ${RESULT_JSON}"
