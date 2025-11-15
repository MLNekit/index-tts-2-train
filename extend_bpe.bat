uv run tools/tokenizer/extend_bpe.py `
    --base-model checkpoints/bpe.model `
    --manifests datasets/italian_emilia_dataset/italian_transcribed.jsonl `
    --output-model checkpoints/italian_bpe.model `
    --target-size 24000 `
    --character-coverage 1.0 `
    --model-type bpe