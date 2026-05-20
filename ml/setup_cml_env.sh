#!/usr/bin/env bash
# Set up cml_env for CoreML export (requires sklearn ≤ 1.5.1)
# Run from project root: bash ml/setup_cml_env.sh
set -e
python3 -m venv cml_env
cml_env/bin/pip install --upgrade pip
cml_env/bin/pip install scikit-learn==1.5.1 coremltools numpy pandas
echo "Done. To export CoreML:"
echo "  cml_env/bin/python ml/coreml_convert.py --notes '설명'"
