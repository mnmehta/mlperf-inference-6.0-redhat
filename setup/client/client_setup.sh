 git clone --recurse-submodule https://github.com/openshift-psap/mlperf-inference-6.0-redhat.git
 uv venv -p 3.12 gptoss_harness
 source gptoss_harness/bin/activate
 cd mlperf-inference-6.0-redhat/language/gpt-oss-120b
 uv pip install pip
 ./setup.sh
 uv pip install transformers==4.57.6
 uv pip install mlflow boto3

