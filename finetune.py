# Adapted from https://github.com/artidoro/qlora/blob/main/qlora.py

from datetime import datetime
import json
import logging
import os

from peft import (
    get_peft_model, 
    LoraConfig, 
    PeftModel,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Seq2SeqTrainer,
)
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

from arguments_schema import DataArgs, ModelArgs, TrainArgs
from dataset_utils import get_data_module


logging.basicConfig(
    filename=f"finetune_{datetime.today().strftime('%Y-%m-%d %H:%M:%S')}.log",
    format="%(asctime)s {%(pathname)s:%(lineno)d} %(name)s %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)


def get_last_checkpoint(output_dir: str):
    """Retrieves the last checkpoint directory."""
    if not os.path.isdir(output_dir):
        return None # first training
    
    if os.path.exists(os.path.join(output_dir, "completed")): 
        logger.info('Detected that training was already completed!')
        return None
    
    max_step = 0
    for filename in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, filename)) and filename.startswith("checkpoint"):
            max_step = max(max_step, int(filename[len("checkpoint-"):]))
    if max_step == 0:
        return None # training started, but no checkpoint
    
    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{max_step}')
    logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")

    return checkpoint_dir # checkpoint found!


def get_model(model_args: ModelArgs, train_args: TrainArgs, checkpoint_dir: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=train_args.bits == 4,
        load_in_8bit=train_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_args.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN")
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, "adapter_model"), is_trainable=True)
    else:
        logger.info("Adding LoRA modules.")
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_args.lora_r,
            lora_alpha=train_args.lora_alpha,
            lora_dropout=train_args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, config)

    model.config.use_cache = False
    model.print_trainable_parameters()
    logger.info("Loaded model.")

    return model


def prepare_llama2(model: LlamaForCausalLM, tokenizer: LlamaTokenizer):
    # Vocab_size mismatch
    # https://github.com/huggingface/transformers/issues/24899

    assert tokenizer.vocab_size == model.vocab_size
    assert tokenizer.get_added_vocab() == {"<pad>": 32000}

    tokenizer.add_special_tokens(dict(pad_token="<pad>"))

    # Update model's embeddings data to include pad_token
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings_data = model.get_input_embeddings().weight.data
    output_embeddings_data = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings_data[:-1].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings_data[:-1].mean(dim=0, keepdim=True)

    input_embeddings_data[-1:] = input_embeddings_avg
    output_embeddings_data[-1:] = output_embeddings_avg

    assert len(tokenizer) == model.vocab_size


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info("Saving PEFT checkpoint...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, **kwargs):
        fname = os.path.join(args.output_dir, "completed")
        with open(fname, "a"):
            os.utime(fname)

        self.save_model(args, state, kwargs)


def train():
    hfparser = transformers.HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    model_args, data_args, train_args = hfparser.parse_args_into_dataclasses()
    
    checkpoint_dir = get_last_checkpoint(train_args.output_dir)

    model = get_model(model_args, train_args, checkpoint_dir)
    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.base_model,
        padding_side="right",
        use_fast=False,
        use_auth_token=os.getenv("HF_ACCESS_TOKEN")
    )
    prepare_llama2(model, tokenizer)

    data_module = get_data_module(data_args, train_args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        **data_module,
    )
    trainer.add_callback(SavePeftModelCallback)

    all_metrics = dict(run_name=train_args.run_name)

    if train_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_metrics
        all_metrics.update(metrics)

    if train_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if train_args.do_train or train_args.do_eval:
        with open(os.path.join(train_args.output_dir, "metrics.json"), "w") as f:
            f.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()