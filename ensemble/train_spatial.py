"""
Training script for Spatial Model
Independent training with its own optimizer and loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import sys

# Ensure we can import from the current directory
sys.path.append(str(Path(__file__).parent))

from spatial_model import SpatialModel
from dataset import create_dataloaders


def save_checkpoint(model, optimizer, epoch, train_loss, val_acc, path):
    """Helper to save checkpoints consistently"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, path)


def train_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        frames = batch['frames'].to(device)  # [B, T, C, H, W]
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            output = model(frames)
            logits = output['logits']
            loss = criterion(logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            frames = batch['frames'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                output = model(frames)
                logits = output['logits']
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train Spatial Model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing videos')
    parser.add_argument('--train-list', type=str, required=True, help='Text file with train video paths and labels')
    parser.add_argument('--val-list', type=str, required=True, help='Text file with val video paths and labels')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data lists
    def load_data_list(list_file):
        """Load video paths and labels from text file"""
        video_paths = []
        labels = []
        with open(list_file, 'r') as f:
            for line in f:
                # FIXED: safer split in case path has spaces (though unlikely)
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    video_paths.append(parts[0])
                    labels.append(int(parts[1]))
        return video_paths, labels
    
    print("Loading data lists...")
    train_videos, train_labels = load_data_list(args.train_list)
    val_videos, val_labels = load_data_list(args.val_list)
    print(f"Train samples: {len(train_videos)}")
    print(f"Val samples: {len(val_videos)}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    # FIXED: Argument names now match the updated dataset.py
    train_loader, val_loader = create_dataloaders(
        train_paths=train_videos,
        train_labels=train_labels,
        val_paths=val_videos,
        val_labels=val_labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames
    )
    
    # Initialize model
    print("Initializing Spatial Model...")
    model = SpatialModel(num_classes=2, embedding_dim=512)
    model = model.to(args.device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0)
            print(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        else:
            print(f"⚠️ Checkpoint {args.resume} not found! Starting from scratch.")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Early stopping set to: {args.patience} epochs")
    print("=" * 80)
    
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, scaler, args.device, epoch + 1
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, args.device)
            
            # Update scheduler
            scheduler.step()
            
            # Log results
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Check for improvement (Early Stopping)
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                patience_counter = 0  # Reset patience
                # Save Best Model
                save_checkpoint(model, optimizer, epoch, train_loss, val_acc, 
                              checkpoint_dir / 'spatial_model_best.pth')
                print(f"  ★ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            
            # Save Latest Model (for resuming)
            save_checkpoint(model, optimizer, epoch, train_loss, val_acc, 
                          checkpoint_dir / 'spatial_model_latest.pth')
            
            # Save Periodic Checkpoint (Every 5 epochs)
            if (epoch + 1) % 5 == 0:
                save_checkpoint(model, optimizer, epoch, train_loss, val_acc, 
                              checkpoint_dir / f'spatial_model_epoch_{epoch+1}.pth')
                print(f"  Snapshot saved at epoch {epoch+1}")
            
            # Trigger Early Stopping
            if patience_counter >= args.patience:
                print(f"\n\n🛑 Early stopping triggered! No improvement for {args.patience} epochs.")
                print(f"Best Accuracy: {best_val_acc:.2f}%")
                break
                
    except KeyboardInterrupt:
        print("\n\n⚠️  INTERRUPTED BY USER (Ctrl+C)")
        print("Saving emergency checkpoint...")
        
        save_checkpoint(model, optimizer, epoch, train_loss if 'train_loss' in locals() else 0.0, 
                      val_acc if 'val_acc' in locals() else 0.0, 
                      checkpoint_dir / 'spatial_model_interrupted.pth')
        
        print(f"Saved to {checkpoint_dir}/spatial_model_interrupted.pth")
        print("You can resume training using --resume checkpoints/spatial_model_interrupted.pth")
        sys.exit(0)
    
    # Save final model
    save_checkpoint(model, optimizer, args.epochs-1, train_loss if 'train_loss' in locals() else 0.0, 
                   val_acc if 'val_acc' in locals() else 0.0, 
                   checkpoint_dir / 'spatial_model.pth')
    
    # Save training history
    with open(checkpoint_dir / 'spatial_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {checkpoint_dir / 'spatial_model.pth'}")


if __name__ == '__main__':
    main()