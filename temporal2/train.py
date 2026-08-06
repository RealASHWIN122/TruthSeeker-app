import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Updated to use the new image-folder dataset
from dataset import PreExtractedTemporalDataset 
from tmodel import TemporalPhysicsEngine

def save_checkpoint(state, dir_path, filename):
    filepath = os.path.join(dir_path, filename)
    if "best" in filename or "last" in filename or "emergency" in filename:
        print(f"   💾 Saving checkpoint to {filepath}")
    torch.save(state, filepath)

def save_training_plots(history, val_labels, val_preds, epoch, save_dir):
    """Generates and saves the Loss/AUC curves and Confusion Matrix as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 1. PLOT LOSS & AUC CURVES ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='Train Loss', marker='o', color='blue')
    plt.plot(range(1, len(history['val_loss']) + 1), history['val_loss'], label='Val Loss', marker='o', color='red')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(history['train_auc']) + 1), history['train_auc'], label='Train AUC', marker='o', color='blue')
    plt.plot(range(1, len(history['val_auc']) + 1), history['val_auc'], label='Val AUC', marker='o', color='red')
    plt.title('Training & Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()
    
    # --- 2. PLOT CONFUSION MATRIX ---
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real', 'Predicted Fake'], 
                yticklabels=['Actual Real', 'Actual Fake'],
                annot_kws={"size": 14})
    plt.title(f'Validation Confusion Matrix (Epoch {epoch})')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"   📈 Graphs updated and saved to {save_dir}")

def train(args):
    # --- SETUP CHECKPOINT DIRECTORY ---
    checkpoint_dir = os.path.join(args.data_dir, "checkpoints_temporal")
    plots_dir = os.path.join(checkpoint_dir, "plots") 
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    last_ckpt_path = os.path.join(checkpoint_dir, "temporal_last_checkpoint.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # ==========================================
    # 1. SETUP PIPELINE (3D Tensors)
    # ==========================================
    print("🧊 Initializing Pre-Extracted 3D Temporal Datasets...")
    
    spatial_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    # Hooking up the new Dataset class
    train_dataset = PreExtractedTemporalDataset(root_dir=train_dir, num_frames=args.num_frames, transform=spatial_transform, is_training=True)
    val_dataset = PreExtractedTemporalDataset(root_dir=val_dir, num_frames=args.num_frames, transform=spatial_transform, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # 2. MODEL, OPTIMIZER & SCHEDULER
    # ==========================================
    model = TemporalPhysicsEngine(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    # Updated GradScaler initialization for newer PyTorch versions
    scaler = torch.amp.GradScaler('cuda')

    best_auc = 0.0
    patience_counter = 0
    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}

    # --- AUTO-RESUME LOGIC ---
    resume_path = args.resume
    if not resume_path and not args.pretrained_model and os.path.isfile(last_ckpt_path):
        print(f"🔄 Found previous run at '{last_ckpt_path}'. Auto-resuming...")
        resume_path = last_ckpt_path

    if resume_path and os.path.isfile(resume_path):
        print(f"📂 Resuming from checkpoint '{resume_path}'")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint.get('best_auc', 0.0)
        patience_counter = checkpoint.get('patience_counter', 0)
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"✅ Successfully resumed from Epoch {start_epoch}. Best AUC so far: {best_auc:.4f}")

    # ==========================================
    # 3. TRAINING LOOP
    # ==========================================
    print(f"\n🚀 Starting STREAM B (Temporal) training for {args.epochs} epochs...")
    print(f"   Physical Batch: {args.batch_size} | Effective Batch: {args.batch_size * args.accumulate_steps}")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_train_loss = 0.0
            train_all_preds = []
            train_all_labels = []

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
            optimizer.zero_grad()

            for i, (videos, labels) in enumerate(train_pbar):
                videos = videos.to(device)
                labels = labels.to(device).float().view(-1)

                with torch.amp.autocast('cuda'):
                    outputs = model(videos).view(-1)
                    loss = criterion(outputs, labels)
                    loss = loss / args.accumulate_steps 

                scaler.scale(loss).backward()
                
                if (i + 1) % args.accumulate_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                actual_loss = loss.item() * args.accumulate_steps
                epoch_train_loss += actual_loss * videos.size(0)
                
                with torch.no_grad():
                    preds_prob = torch.sigmoid(outputs).detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    train_all_preds.extend(preds_prob)
                    train_all_labels.extend(labels_np)

                if i % 5 == 0:
                    preds_bin = (preds_prob > 0.5).astype(int)
                    batch_acc = accuracy_score(labels_np, preds_bin)
                    batch_auc = roc_auc_score(labels_np, preds_prob) if len(np.unique(labels_np)) > 1 else 0.0
                    current_lr = optimizer.param_groups[0]['lr']
                    train_pbar.set_postfix({
                        'Loss': f"{actual_loss:.4f}", 'Acc': f"{batch_acc:.4f}", 
                        'AUC': f"{batch_auc:.4f}", 'LR': f"{current_lr:.6f}"
                    })

            epoch_train_loss = epoch_train_loss / len(train_loader.dataset)
            train_preds_bin = (np.array(train_all_preds) > 0.5).astype(int)
            epoch_train_acc = accuracy_score(train_all_labels, train_preds_bin)
            try:
                epoch_train_auc = roc_auc_score(train_all_labels, train_all_preds)
            except ValueError:
                epoch_train_auc = 0.0

            # --- VALIDATION PHASE ---
            model.eval()
            epoch_val_loss = 0.0
            val_all_labels = []
            val_all_preds = []
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
            
            with torch.no_grad():
                for videos, labels in val_pbar:
                    videos = videos.to(device)
                    labels = labels.to(device).float().view(-1)
                    
                    with torch.amp.autocast('cuda'):
                        outputs = model(videos).view(-1)
                        val_loss = criterion(outputs, labels)
                    
                    epoch_val_loss += val_loss.item() * videos.size(0)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    
                    val_all_labels.extend(labels.cpu().numpy())
                    val_all_preds.extend(probs)
                    val_pbar.set_postfix({'Loss': f"{val_loss.item():.4f}"})
            
            epoch_val_loss = epoch_val_loss / len(val_loader.dataset)
            val_all_labels = np.array(val_all_labels)
            val_all_preds = np.array(val_all_preds)
            val_preds_bin = (val_all_preds > 0.5).astype(int)

            try:
                epoch_val_auc = roc_auc_score(val_all_labels, val_all_preds)
                epoch_val_acc = accuracy_score(val_all_labels, val_preds_bin)
                val_prec = precision_score(val_all_labels, val_preds_bin, zero_division=0)
                val_rec = recall_score(val_all_labels, val_preds_bin, zero_division=0)
                val_f1 = f1_score(val_all_labels, val_preds_bin, zero_division=0)
            except ValueError:
                epoch_val_auc, epoch_val_acc, val_prec, val_rec, val_f1 = 0.0, 0.0, 0.0, 0.0, 0.0

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_auc'].append(epoch_train_auc)
            history['val_auc'].append(epoch_val_auc)
            
            save_training_plots(history, val_all_labels, val_preds_bin, epoch + 1, plots_dir)

            print(f"\n📊 EPOCH [{epoch+1}/{args.epochs}] RESULTS")
            print(f"   Train | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} | AUC: {epoch_train_auc:.4f}")
            print(f"   Val   | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} | AUC: {epoch_val_auc:.4f}")
            print(f"   Ext   | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

            scheduler.step(epoch_val_auc)

            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_auc': best_auc,
                'patience_counter': patience_counter,
                'history': history 
            }

            save_checkpoint(state, checkpoint_dir, f"temporal_checkpoint_epoch_{epoch+1}.pth")

            # --- EARLY STOPPING & CHECKPOINT LOGIC ---
            if epoch_val_auc > best_auc:
                print(f"   🔥 New Best Temporal Model! (AUC: {best_auc:.4f} -> {epoch_val_auc:.4f})")
                best_auc = epoch_val_auc
                patience_counter = 0 
                state['best_auc'] = best_auc
                state['patience_counter'] = 0
                save_checkpoint(state, checkpoint_dir, "temporal_best_model.pth")
            else:
                patience_counter += 1
                state['patience_counter'] = patience_counter
                print(f"   📉 No improvement. Patience: {patience_counter}/{args.patience}")

            save_checkpoint(state, checkpoint_dir, "temporal_last_checkpoint.pth")
            print("-" * 60)

            if patience_counter >= args.patience:
                print(f"\n🛑 Early Stopping triggered! No temporal improvement for {args.patience} epochs.")
                break

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted! Saving emergency 3D checkpoint...")
        state = {
            'epoch': epoch if 'epoch' in locals() else 0, 
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict() if 'scheduler' in locals() else None,
            'best_auc': best_auc,
            'patience_counter': patience_counter,
            'history': history if 'history' in locals() else None
        }
        save_checkpoint(state, checkpoint_dir, "temporal_emergency_checkpoint.pth")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to processed temporal dataset (folders of frames)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate_steps", type=int, default=8, help="Steps to accumulate gradients")
    parser.add_argument("--num_frames", type=int, default=16, help="Timeline depth for the 3D tensor")
    parser.add_argument("--lr", type=float, default=1e-4) 
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait before early stopping")
    parser.add_argument("--resume", type=str, default=None, help="Force resume from specific checkpoint")
    parser.add_argument("--pretrained_model", type=str, default=None, help="Path to a custom Foundation Model")
    
    args = parser.parse_args()
    train(args)