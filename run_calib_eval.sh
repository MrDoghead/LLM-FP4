export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_ADDR=/data/huggyllama/llama-7b

python main.py --model hf-causal-experimental \
	--model_args pretrained=$MODEL_ADDR,use_accelerate=True \
	--tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande \
	--device cuda \
	--batch_size auto \
	--no_cache \
	--num_fewshot 0 \
	--quant_config 'FPQ_config_llama' \
	--qbits 4 4 4 2 2 2 \
	--calib_size 32 \
	--search_round 3 \
	--search_intervals 0.01 1.2 100 \
	--output_path /workspace/LLM-FP4/logs/run_calib_eval_results.json \
	--write_out \
	--output_base_path /workspace/LLM-FP4/logs/ \
	--ptq_param_path "./q_params/FP4_FPQ.pt" \

#--limit 128 \
#--search_intervals 0.01 1.2 100 \
