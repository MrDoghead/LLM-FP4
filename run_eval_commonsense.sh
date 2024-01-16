export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_ADDR=/data/models/huggyllama/llama-7b

python main.py --model hf-causal-experimental \
	--model_args pretrained=$MODEL_ADDR,use_accelerate=True \
	--tasks arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande \
	--limit 0.1 \
	--device cuda \
	--batch_size auto \
	--no_cache \
	--num_fewshot 0 \
	--quant_config 'FPQ_config_llama' \
	--qbits 4 4 4 2 2 2 \
	--only_eval \
	--ptq_param_path "/data/models/LLM-FP4/search_result/FPQ_config_llama/W4A4E4_search_round3_search_intervals-0.01-1.2-100.0.pt"

#--ptq_param_path "./search_result/FPQ_config_llama/W4A4E4_search_round3_search_intervals(0.01,1.2,100).pt"
