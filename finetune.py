# Adapted from https://github.com/artidoro/qlora/blob/main/qlora.py and https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

from datetime import datetime
import json
import logging
import os

from transformers import Seq2SeqTrainer
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from arguments_schema import DataArgs, ModelArgs, TrainArgs
from dataset_utils import get_data_module
from model_utils import get_model_and_tokenizer


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

    def on_train_end(self, args, state, control, **kwargs):
        fname = os.path.join(args.output_dir, "completed")
        with open(fname, "a"):
            os.utime(fname)

        self.save_model(args, state, kwargs)


def train():
    hfparser = transformers.HfArgumentParser((ModelArgs, DataArgs, TrainArgs))
    model_args, data_args, train_args = hfparser.parse_args_into_dataclasses()
    
    checkpoint_dir = get_last_checkpoint(train_args.output_dir)
    model, tokenizer = get_model_and_tokenizer(model_args, checkpoint_dir)
    data_module = get_data_module(data_args, tokenizer)

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
    try:
        train()
    except Exception as e:
        logger.error(e)
        raise e