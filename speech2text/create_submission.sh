python3 ../compliance/TEST01/run_verification.py -r /data/inference/speech2text/v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/ -c /data/inference/speech2text/logs_compliance/ -o /data/inference/speech2text/v6_submission_proper/closed/RedHat/results/H200/whisper/Offline >& output.txt 
bash ../compliance/TEST01/\create_accuracy_baseline.sh v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/accuracy/mlperf_log_accuracy.json  v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/TEST01/accuracy/mlperf_log_accuracy.json 
Created a baseline accuracy file: mlperf_log_accuracy_baseline.json
cp mlperf_log_accuracy_baseline.json mlperf_log_accuracy.json 
python3 accuracy_eval.py --log_dir ./ --dataset_dir . --manifest data/dev-all-repack.json >&  v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/TEST01/accuracy/baseline_accuracy.txt
python3 accuracy_eval.py --log_dir v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/TEST01/accuracy/ --dataset_dir . --manifest data/dev-all-repack.json >& v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/TEST01/accuracy/compliance_accuracy.txt 
cp v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/TEST01/accuracy/compliance_accuracy.txt v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/TEST01/accuracy/accuracy.txt 
cp README.md v6_submission_proper/closed/RedHat/results/H200/whisper/Offline/
python3 ../tools/submission/truncate_accuracy_log.py  --input v6_submission_proper  --submitter RedHat  --output v6_submission_proper/truncated
python3 ../tools/submission/submission_checker/main.py  --input v6_submission_proper/truncated/  --version v6.0 --debug
