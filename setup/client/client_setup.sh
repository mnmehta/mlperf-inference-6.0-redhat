 # Install system dependencies
 apt-get update -qq && apt-get install -y -qq git build-essential curl > /dev/null 2>&1
 pip install -q uv

 git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.0-redhat.git
 uv venv -p 3.12 gptoss_harness
 source gptoss_harness/bin/activate
 cd mlperf-inference-6.0-redhat/language/gpt-oss-120b
 uv pip install pip
 ./setup.sh

 # Install harness-specific dependencies (includes transformers, mlflow, boto3)
 cd ../../harness
 pip install -r requirements.txt

