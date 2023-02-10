# Multilingual Robust Constrastive Pretraining

This code is released as part of our EACL paper on Robustification of Multilingual Language Models to Real-world Noise with Robust Contrastive Pretraining (official link coming up).

## Citation

If you use code/data in this repository, you will have to cite the following work:

```
@proceedings{eacl-2023-asa-sailik,
    title = "Robustification of Multilingual Language Models to Real-world Noise with Robust Contrastive Pretraining",
    author = {Stickland, Asa Cooper and Sengupta, Sailik and Krone, Jason and Mansour, Saab and He, He},
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    year = "2023",
    publisher = "Association for Computational Linguistics"
}
```

## Dependencies

A [conda\_env.yml](./conda_env.yml) can be found here (depending on the code you run, you may need additional dependencies).

## LICENSES
The code base is build on the shoulder of other code-bases. Licenses for these code bases can be found inside [THIRD\_PARTY\_LICENSES.md](./THIRD_PARTY_LICENSES.md). And amendment made to the code is licenses as per [LICENSE](./LICENSE).

The data inside [paper\_data](./paper_data) have licenses of their own. More information about it can found in [this README.md](./paper_data/README.md) file.

## Training & Evaluation

Example commands
```bash
# For IC-SL datasets (eg. SNIPS)
python main.py \
    --task ${DATASETS} (eg. snips)\
    --model_type ${MODEL} (eg. "xlmr")\
    --num_train_epochs ${EPOCH} \
    --seed ${seed} \
    --max_seq_len ${seq_len} \
    --learning_rate ${LR_RATE} \
    --train_lang ${TRAIN_LANGUAGE} (eg. "en")\
    --dev_lang ${DEV_LANGUAGE} (eg. "fr")\
    --predict_languages ${TEST_LANGUAGE} (eg. "fr,fr_typos_0.1") \
    --data_dir ${dir_where_data_is_saved} (eg ./paper_data/snips) \
    --model_dir ${output_dir} \
    --reset_cache \
    --do_train \
    --do_eval \
    --write_preds
    
# for NLI datasets (Eg. XNLI)
python pretraining/run_classify.py \
    --model_name_or_path ${MODEL} (eg. xlm-roberta-base) \
    --train_dir ${training_data_dir} (eg ./paper_data/xnli) \
    --predict_langs ${TEST_LANGUAGE} \
    --noise_types ${NOISE_TYPES} \
    --train_language ${TRAIN_LANGUAGE} \
    --language ${DEV_LANGUAGE} \
    --do_predict \
    --do_train \
    --save_steps {eg. 10000} \
    --max_seq_length {eg. 128} \
    --per_device_train_batch_size {eg. 32} \
    --learning_rate ${LR_RATE} \
    --num_train_epochs ${EPOCH} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --seed $seed
    
# for NER datasets (Eg. WikiANN)
python pretraining/run_ner.py \
    --model_name_or_path $MODEL \
    --train_dir ${training_data_dir} (eg ./paper_data/panx) \
    --predict_langs ${TEST_LANGUAGE} \
    --noise_types ${NOISE_TYPES} \
    --train_language ${TRAIN_LANGUAGE} \
    --language ${DEV_LANGUAGE} \
    --do_predict \
    --save_steps {eg. 10000} \
    --max_seq_length {eg. 128} \
    --per_device_train_batch_size {eg. 32} \
    --per_device_eval_batch_size {eg. 128} \
    --learning_rate ${LR_RATE} \
    --num_train_epochs ${EPOCH} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --do_train \
    --seed $seed
```

Run scripts can be found inside the [runner_scripts](./runner_scirpts) directory.
