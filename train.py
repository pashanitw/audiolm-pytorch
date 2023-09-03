import torch
from pathlib import Path
from typing import Union, Tuple, Optional
from einops import rearrange, reduce
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import csv
from beartype import beartype
from beartype.typing import Tuple, Union, Optional
from beartype.door import is_bearable
from audiolm_pytorch import HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer,\
    CoarseTransformer, CoarseTransformerTrainer, FineTransformer, FineTransformerTrainer, EncodecWrapper
from audiolm_pytorch.data import cast_tuple
from torchaudio.functional import resample
import argparse

class SoundDataset(Dataset):
    @beartype
    def __init__(
        self,
        folder,
        target_sample_hz: Union[int, Tuple[int, ...]],
        exts=['flac', 'wav', 'mp3', 'webm'],
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        self.metadata_path = path / 'metadata.csv'
        assert self.metadata_path.exists(), 'metadata.csv does not exist'

        skip_files = []
        # Parsing the metadata.csv file to get text transcriptions
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            self.metadata = {}
            for row in reader:
                if len(row) == 3:
                    self.metadata[row[0]] = row[2]
                else:
                    skip_files.append(row[0])


        # Audio files
        files = [file for ext in exts for file in path.glob(f'wavs/*.{ext}')]
        files = [file for file in files if file.stem not in skip_files]
        assert len(files) > 0, 'no sound files found'

        self.files = files
        self.target_sample_hz = cast_tuple(target_sample_hz)

        self.max_target_sample_hz = max(self.target_sample_hz)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file_id = file.stem

        # Ensure we have a transcription for this file
        assert file_id in self.metadata, f'No transcription found for {file_id}'

        transcription = self.metadata[file_id]

        data, sample_hz = torchaudio.load(file)

        assert data.numel() > 0, f'one of your audio file ({file}) is empty. please remove it from your folder'

        if data.shape[0] > 1:
            # the audio has more than 1 channel, convert to mono
            data = reduce(data, 'c ... -> 1 ...', 'mean')

        # resample data to the max target freq
        data = resample(data, sample_hz, self.max_target_sample_hz)
        sample_hz = self.max_target_sample_hz

        data = rearrange(data, '1 ... -> ...')
        # print("==========min and max of data==========", data.min(), data.max())
        return data, sample_hz, self.target_sample_hz, transcription

def collate_fn(batch):
    # List of sequences and their corresponding sampling rates
    sequences, sample_hz_list, target_sample_hz_list, transcription = zip(*batch)
    # print(sequences[0].shape, sequences[1].shape)

    # Determine max length within this batch
    max_length = max([seq.size(0) for seq in sequences])

    # Pad sequences to this max length
    sequences_padded = [F.pad(seq, (0, max_length - seq.size(0)), 'constant') for seq in sequences]

    # Stack sequences into a tensor
    sequences_tensor = torch.stack(sequences_padded)

    # print(sequences_tensor.shape)
    # print("==========min and max of data==========", sequences_tensor.min(), sequences_tensor.max())

    # convert tuple to list
    transcription = list(transcription)

    return transcription,sequences_tensor


# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert/hubert_base_ls960.pt',
    kmeans_path = './hubert/hubert_base_ls960_L9_km500.bin'
)
encodec = EncodecWrapper()
semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6,
    flash_attn = False,
    has_condition=True,
    cond_as_self_attn_prefix=True,
)

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6,
    flash_attn = False
)


fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6,
    flash_attn = True
)

dataset = SoundDataset(
    folder = './LJSpeech-1.1',
    target_sample_hz=wav2vec.target_sample_hz,
)

def main(args):
    model_type = args.type
    batch_size = args.batch_size
    steps = args.steps

    trainer = None

    if model_type == 'semantic':
        trainer = SemanticTransformerTrainer(
            transformer=semantic_transformer,
            wav2vec=wav2vec,
            dataset=dataset,
            batch_size=batch_size,
            num_train_steps=steps,
            collate_fn=collate_fn,
        )

    if model_type == 'coarse':
        trainer = CoarseTransformerTrainer(
            transformer=coarse_transformer,
            codec = encodec,
            wav2vec=wav2vec,
            folder='./LJSpeech-1.1/wavs',
            batch_size=batch_size,
            num_train_steps=steps,
            grad_accum_every=4
        )
    if model_type == 'fine':
        trainer = FineTransformerTrainer(
            transformer=fine_transformer,
            codec=encodec,
            folder='./LJSpeech-1.1/wavs',
            batch_size=batch_size,
            num_train_steps=steps,
            data_max_length=320*64
        )


    if trainer is not None:
        trainer.train()

    # ... Rest of your training code ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for audio model")

    # Adding argument for type
    parser.add_argument('--type', type=str, required=True,
                        choices=['semantic', 'coarse', 'fine'],  # You can list the valid model types here
                        help="Specify the type of the model to be trained")

    # Adding argument for batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify the batch size for training")

    # Adding argument for steps
    parser.add_argument('--steps', type=int, default=1000,
                        help="Specify the number of training steps")

    args = parser.parse_args()

    if(args.type == 'semantic'):
        print("Training semantic model")

    main(args)
