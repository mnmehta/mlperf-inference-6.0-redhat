DIR=$1
TEST=$2
 python3 ../compliance/${TEST}/run_verification.py -c ${DIR}/mlperf/ -o ${DIR} \
    --audit-config ../compliance/${TEST}/gpt-oss-120b/audit.config \
    --accuracy-script "python3 ../language/gpt-oss-120b/eval_mlperf_accuracy.py \
        --mlperf-log ${DIR}/mlperf/mlperf_log_accuracy.json \
        --reference-data ${DATASET_DIR}/acc/acc_eval_compliance_gpqa.parquet  \
        --tokenizer openai/gpt-oss-120b"
