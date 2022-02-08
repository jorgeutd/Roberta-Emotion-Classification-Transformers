from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,roc_auc_score
from datasets import load_from_disk
import random
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    
        
#     # Push to Hub Parameters
#     parser.add_argument("--push_to_hub", type=bool, default=True)
#     parser.add_argument("--hub_model_id", type=str, default=None)
#     parser.add_argument("--hub_strategy", type=str, default=None)
#     parser.add_argument("--hub_token", type=str, default=None)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_VAL"])
    

    args, _ = parser.parse_known_args()
    
    os.environ['GPU_NUM_DEVICES']=args.n_gpus

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    
#     # make sure we have required parameters to push
#     if args.push_to_hub:
#         if args.hub_strategy is None:
#             raise ValueError("--hub_strategy is required when pushing to Hub")
#         if args.hub_token is None:
#             raise ValueError("--hub_token is required when pushing to Hub")

#     # sets hub id if not provided
#     if args.hub_model_id is None:
#         args.hub_model_id = args.model_id.replace("/", "--")

    # load datasets
    train_dataset = load_from_disk(args.training_dir)
    val_dataset = load_from_disk(args.val_dir)

    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(val_dataset)}")
    
     # Prepare model labels - useful in inference API
    labels = train_dataset.features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # compute metrics function for multiclass classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    


    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="f1"
#         # push to hub parameters
#         push_to_hub=args.push_to_hub,
#         hub_strategy=args.hub_strategy,
#         hub_model_id=args.hub_model_id,
#         hub_token=args.hub_token,
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=val_dataset)

#     # save best model, metrics and create model card
#     trainer.create_model_card(model_name=args.hub_model_id)
#     trainer.push_to_hub()

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(os.environ["SM_MODEL_DIR"])


    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # Saves the model to s3
    trainer.save_model(os.environ["SM_MODEL_DIR"])
