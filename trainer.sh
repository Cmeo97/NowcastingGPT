PYTHON_SCRIPT="src/main.py"
CONFIG_PATH="/____/____/NowcastingGPT_new/config"
CONFIG_NAME="trainer.yaml"
CONFIG_FILE="${CONFIG_PATH}/${CONFIG_NAME}"

sed -i '/training:/,/^  [^ ]/s/^\(\s*should:\s*\).*/\1True/' $CONFIG_FILE
sed -i '/evaluation:/,/^  [^ ]/s/^\(\s*should:\s*\).*/\1False/' $CONFIG_FILE

nohup python -u $PYTHON_SCRIPT \
  --config-path $CONFIG_PATH \
  --config-name $CONFIG_NAME > nohup.out 2>&1 &
    

