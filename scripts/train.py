import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from model import ProteinMPNNModel, get_device
from data import ProteinDataModule

"""
Usage:
python train.py --path_for_training_data ../data/pdb_2021aug02_sample --path_for_outputs ./logs
"""

def main(args):
    # Get device
    device = get_device()
    # Force CPU if MPS memory error occurs
    if os.environ.get('FORCE_CPU'):
        accelerator = 'cpu'
        print("Using CPU as requested by environment variable")
    else:
        accelerator = 'gpu' if str(device) in ['cuda:0', 'mps'] else 'cpu'
    
    # Initialize data module with smaller batch size if using MPS
    effective_batch_size = args.batch_size
    if str(device) == 'mps':
        # Reduce batch size for MPS and use gradient accumulation
        actual_batch_size = 1  # Very small batch size for MPS
        accumulate_grad_batches = effective_batch_size
        print(f"MPS detected: Using batch_size={actual_batch_size} with gradient accumulation={accumulate_grad_batches}")
    else:
        actual_batch_size = effective_batch_size
        accumulate_grad_batches = args.accumulate_grad_batches

    data_module = ProteinDataModule(
        data_dir=args.path_for_training_data,
        batch_size=actual_batch_size,
        num_workers=args.num_workers,
        rescut=args.rescut,
        debug=args.debug
    )

    # Initialize model
    model = ProteinMPNNModel(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_epochs=args.max_epochs
    )

    # Load from checkpoint if provided
    if args.previous_checkpoint:
        model = ProteinMPNNModel.load_from_checkpoint(args.previous_checkpoint)

    # Setup logging
    logger = TensorBoardLogger(
        save_dir=args.path_for_outputs,
        name='protein_mpnn',
        default_hp_metric=False
    )

    # Tensorboard usage
    # tensorboard --logdir=./logs/protein_mpnn
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.path_for_outputs, 'checkpoints'),
            filename='protein_mpnn-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]

    # Initialize trainer with modified settings
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip,
        precision=16 if args.use_amp else 32,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=args.val_check_interval
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--path_for_training_data', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--path_for_outputs', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--previous_checkpoint', type=str,
                        help='Path to previous checkpoint to resume training')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num_encoder_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=3,
                        help='Number of decoder layers')
    parser.add_argument('--num_neighbors', type=int, default=32,
                        help='Number of neighbors for attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--backbone_noise', type=float, default=0.1,
                        help='Backbone noise for data augmentation')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                        help='Number of warmup steps')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Number of batches to accumulate gradients')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='Validation check interval')
    
    # Hardware arguments
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA if available')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    
    # Other arguments
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    parser.add_argument('--rescut', type=float, default=3.5,
                        help='Resolution cutoff for PDBs')
    
    args = parser.parse_args()
    main(args) 