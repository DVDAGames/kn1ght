from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

TRAINING_DATA_PATH = "./.data/training_data.txt"
OUTPUT_PATH = "./src/tokenizer"

with open(TRAINING_DATA_PATH, "r") as f:
    lines = f.readlines()

    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]", fuse_unk=True, continuing_subword_prefix="")
    )

    tokenizer.normalizer = NFD()

    tokenizer.pre_tokenizer = Split(
        pattern=Regex(r""" ?\d+\.|\. ?| ?[-\w]+|[#+]"""), behavior="isolated"
    )

    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=4096, show_progress=True)

    tokenizer.train_from_iterator(lines, trainer=trainer)

    tokenizer.save(f"{OUTPUT_PATH}/kn1ght-tokenizer.json")

    tokenizer.model.save(OUTPUT_PATH, "kn1ght")
