import argparse

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFD
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer

from kn1ght.constants import SPECIAL_TOKENS

TRAINING_DATA_PATH = "./.data/training_data.txt"
OUTPUT_PATH = "./src/tokenizer"
VOCAB_SIZE = 4096

# get args from command line
parser = argparse.ArgumentParser()

parser.add_argument(
    "--training_data_path",
    help="Path to the training data file",
    default=TRAINING_DATA_PATH,
)

parser.add_argument(
    "--output_path",
    help="Path to save the tokenizer",
    default=OUTPUT_PATH,
)

parser.add_argument(
    "--vocab_size",
    help="Size of the vocabulary",
    default=VOCAB_SIZE,
)

args = parser.parse_args()


with open(args.training_data_path, "r") as f:
    print(f"Training kn1ght tokenizer with {args.training_data_path}...")

    lines = f.readlines()

    lines = [
        SPECIAL_TOKENS["START"] + line.strip() + SPECIAL_TOKENS["END"] for line in lines
    ]

    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["UNKNOWN"], fuse_unk=True))

    tokenizer.normalizer = NFD()

    tokenizer.pre_tokenizer = Split(
        pattern=Regex(r""" ?\d+\.|\. ?| ?[-\w]+|[#+]"""), behavior="isolated"
    )

    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        special_tokens=list(SPECIAL_TOKENS.values()),
    )

    tokenizer.train_from_iterator(lines, trainer=trainer)

    tokenizer.save(f"{args.output_path}/kn1ght-tokenizer.json")

    tokenizer.model.save(args.output_path, "kn1ght")
