import argparse
import random
import copy
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize
from pathlib import Path

# Helper function to save model state
def save_ckpt(model, path, is_distributed):
    """Saves the model checkpoint."""
    # If using DistributedDataParallel, access the underlying model
    model_state = model.module.state_dict() if is_distributed else model.state_dict()
    torch.save(model_state, path)
    print(f"Checkpoint saved to {path}")

def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', default=False, type=bool)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.5, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=60, type=int)
    # Loss coefficients
    parser.add_argument('--teach_box_loss', default=False, type=bool)
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--coef_target', default=1.0, type=float)
    parser.add_argument('--coef_domain', default=1.0, type=float)
    parser.add_argument('--coef_domain_bac', default=0.3, type=float)
    parser.add_argument('--coef_mae', default=1.0, type=float)
    parser.add_argument('--alpha_focal', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.9996, type=float)
    # Dataset parameters
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--source_dataset', default='tanks_source', type=str)
    parser.add_argument('--target_dataset', default='tanks_source', type=str)
    # parser.add_argument('--target_dataset', default='tanks_test', type=str)    
    # parser.add_argument('--target_dataset', default='tanks_source', type=str)
    # Retraining parameters
    parser.add_argument('--epoch_retrain', default=40, type=int)
    parser.add_argument('--keep_modules', default=["decoder"], type=str, nargs="+")
    # MAE parameters
    parser.add_argument('--mae_layers', default=[2], type=int, nargs="+")
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--epoch_mae_decay', default=10, type=float)
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training, "
                             "'cross_domain_mae' for cross domain training with mae, "
                             "'teaching' for teaching process, 'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=8008, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)

    # CHANGED: Modified help text to clarify this is for evaluation only
    parser.add_argument('--eval-from-checkpoint', default=None, type=str,
                        help="Path to the model checkpoint to load for evaluation only. Skips the training phase.")
    
    # NEW: Added argument for continuing training from checkpoint
    parser.add_argument('--continue-from-checkpoint', default=None, type=str,
                        help="Path to the model checkpoint to load and continue training from. "
                             "This will load model weights, optimizer state, and resume from the saved epoch.")
    
    # NEW: Added argument to store starting epoch when continuing training
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="Starting epoch number when continuing training from a checkpoint. "
                             "Automatically set when using --continue-from-checkpoint.")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP50', m_ap, epoch)
    for idx, num in zip(idx_to_class.keys(), ap_per_class):
        writer.add_scalar(prefix + '/AP50_%s' % (idx_to_class[idx]['name']), num, epoch)


def single_domain_training(model, device):
    # Record the start time
    start_time = time.time()
    # Build dataloaders
    # train_loader = build_dataloader(args, args.source_dataset, 'source', 'train', train_trans)
    train_loader = build_dataloader(args, args.source_dataset, 'source', 'train', train_trans)    
    #val_loader = build_dataloader(args, args.target_dataset, 'source', 'val', val_trans)
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    # val_loader = build_dataloader(args, args.source_dataset, 'target', 'val', val_trans)    
    idx_to_class = val_loader.dataset.coco.cats
    # Prepare model for optimization
    # import pdb;pdb.set_trace()
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion = build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    
    # CHANGED: Added logic to handle continuing training from checkpoint
    if args.continue_from_checkpoint:
        print(f"💡 Loading checkpoint from {args.continue_from_checkpoint} to continue training.")
        checkpoint = torch.load(args.continue_from_checkpoint, map_location=device, weights_only=False)
        
        # Load model state
        model_to_load = model.module if args.distributed else model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint includes optimizer and epoch info
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            if 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"✅ Resuming training from epoch {args.start_epoch}")
        else:
            # Checkpoint only contains model weights
            model_to_load.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights. Starting from epoch {args.start_epoch}")
    
    if args.eval_from_checkpoint:
        print(f"💡 Loading model from {args.eval_from_checkpoint} for evaluation only.")
        checkpoint = torch.load(args.eval_from_checkpoint, map_location=device, weights_only=False)
        
        # Handle models saved with/without DistributedDataParallel wrapper
        model_to_load = model.module if args.distributed else model
        # CHANGED: Added support for dict-based checkpoints
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        else:
            model_to_load.load_state_dict(checkpoint)
        print("Model loaded successfully. Skipping training.")

    # Record the best mAP
    ap50_best = -1.0
    # CHANGED: Use args.start_epoch instead of 0 to support continuing training
    for epoch in range(args.start_epoch, args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        if not args.eval_from_checkpoint:

            loss_train, loss_train_dict = train_one_epoch_standard(
                model=model,
                criterion=criterion,
                data_loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                clip_max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                flush=args.flush
            )
            write_loss(epoch, 'single_domain', loss_train, loss_train_dict)
            lr_scheduler.step()
            
            # CHANGED: Save comprehensive checkpoint including optimizer state
            eval_ckpt_path = Path(output_dir) / 'model_for_eval.pth'
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': (model.module if args.distributed else model).state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                # 'args': args
            }
            torch.save(checkpoint_dict, eval_ckpt_path)
            print(f"Checkpoint saved to {eval_ckpt_path}")

        # Evaluate
        # try:
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # except:
        #     import pdb;pdb.set_trace()
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            # CHANGED: Save comprehensive checkpoint for best model too
            best_checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': (model.module if args.distributed else model).state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                # 'map50': map50,
                # 'args': args
            }
            torch.save(best_checkpoint_dict, output_dir/'model_best.pth')
        #if epoch == args.epoch - 1:
            # CHANGED: Save comprehensive checkpoint for last model
        last_checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': (model.module if args.distributed else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'args': args
        }
        torch.save(last_checkpoint_dict, output_dir/'model_last.pth')
        # Write the evaluation results to tensorboard
        print(f"this is map50: {map50}")
        write_ap50(epoch, 'single_domain', map50, ap50_per_class, idx_to_class)
        
        # If in eval-only mode, you might want to break after one loop
        if args.eval_from_checkpoint:
            print("Evaluation finished. Exiting eval-only mode.")
            break        
    # Record the end time
    # python main.py --epoch 1 --eval-from-checkpoint /app/app4/output/model_for_eval.pth

    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Single-domain training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


def cross_domain_mae(model, device):
    start_time = time.time()
    # Build dataloaders
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', strong_trans)
    target_loader = build_dataloader(args, args.target_dataset, 'target', 'train', strong_trans)
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Build MAE branch
    image_size = target_loader.dataset.__getitem__(0)[0].shape[-2:]
    model.transformer.build_mae_decoder(image_size, args.mae_layers, device, channel0=model.backbone.num_channels[0])
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion, criterion_mae = build_criterion(args, device), build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    
    # NEW: Added logic to handle continuing training from checkpoint
    if args.continue_from_checkpoint:
        print(f"💡 Loading checkpoint from {args.continue_from_checkpoint} to continue training.")
        checkpoint = torch.load(args.continue_from_checkpoint, map_location=device, weights_only=False)
        
        model_to_load = model.module if args.distributed else model
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f"✅ Resuming training from epoch {args.start_epoch}")
        else:
            model_to_load.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights. Starting from epoch {args.start_epoch}")
    
    # Record the best mAP
    ap50_best = -1.0
    # CHANGED: Use args.start_epoch instead of 0
    for epoch in range(args.start_epoch, args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train, loss_train_dict = train_one_epoch_with_mae(
            model=model,
            criterion=criterion,
            criterion_mae=criterion_mae,
            source_loader=source_loader,
            target_loader=target_loader,
            coef_target=args.coef_target,
            mask_ratio=args.mask_ratio,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        write_loss(epoch, 'cross_domain_mae', loss_train, loss_train_dict)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.0001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            # CHANGED: Save comprehensive checkpoint
            best_checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': (model.module if args.distributed else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'map50': map50,
                'args': args
            }
            torch.save(best_checkpoint_dict, output_dir/'model_best.pth')
        #if epoch == args.epoch - 1:
        last_checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': (model.module if args.distributed else model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'args': args
        }
        torch.save(last_checkpoint_dict, output_dir/'model_last.pth')
        # Write the evaluation results to tensorboard
        write_ap50(epoch, 'cross_domain_mae', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Cross-domain MAE training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


# Teaching
def teaching(model_stu, device):
    start_time = time.time()
    # Build dataloaders
    source_loader = build_dataloader(args, args.source_dataset, 'source', 'train', strong_trans)
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'train')
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Build teacher model
    model_tch = build_teacher(args, model_stu, device)
    # Build discriminators
    model_stu.build_discriminators(device)
    # Build MAE branch
    image_size = target_loader.dataset.__getitem__(0)[0].shape[-2:]
    model_stu.transformer.build_mae_decoder(image_size, args.mae_layers, device, model_stu.backbone.num_channels[0])
    # Prepare model for optimization
    if args.distributed:
        model_stu = DistributedDataParallel(model_stu, device_ids=[args.gpu], find_unused_parameters=True)
        model_tch = DistributedDataParallel(model_tch, device_ids=[args.gpu])
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device, box_loss=args.teach_box_loss)
    optimizer = build_optimizer(args, model_stu)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    
    # NEW: Added logic to handle continuing training from checkpoint
    if args.continue_from_checkpoint:
        print(f"💡 Loading checkpoint from {args.continue_from_checkpoint} to continue teaching.")
        checkpoint = torch.load(args.continue_from_checkpoint, map_location=device, weights_only=False)
        
        model_stu_load = model_stu.module if args.distributed else model_stu
        model_tch_load = model_tch.module if args.distributed else model_tch
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_stu_load.load_state_dict(checkpoint['model_state_dict'])
            # Also load teacher if saved
            if 'teacher_state_dict' in checkpoint:
                model_tch_load.load_state_dict(checkpoint['teacher_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            # Load thresholds if saved
            if 'thresholds' in checkpoint:
                thresholds = checkpoint['thresholds']
            else:
                thresholds = [args.threshold] * args.num_classes
            print(f"✅ Resuming teaching from epoch {args.start_epoch}")
        else:
            model_stu_load.load_state_dict(checkpoint)
            thresholds = [args.threshold] * args.num_classes
            print(f"✅ Loaded model weights. Starting from epoch {args.start_epoch}")
    else:
        # Initialize thresholds
        thresholds = [args.threshold] * args.num_classes
    
    # Reinitialize checkpoint for selective retraining
    reinit_ckpt = copy.deepcopy(model_tch.state_dict())
    # Record the best mAP
    ap50_best = -1.0
    # CHANGED: Use args.start_epoch instead of 0
    for epoch in range(args.start_epoch, args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(source_loader.sampler, 'set_epoch'):
            source_loader.sampler.set_epoch(epoch)
            target_loader.sampler.set_epoch(epoch)
        loss_train, loss_source_dict, loss_target_dict = train_one_epoch_teaching(
            student_model=model_stu,
            teacher_model=model_tch,
            criterion=criterion,
            criterion_pseudo=criterion_pseudo,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            thresholds=thresholds,
            coef_target=args.coef_target,
            mask_ratio=args.mask_ratio,
            alpha_ema=args.alpha_ema,
            device=device,
            epoch=epoch,
            enable_mae=(epoch < args.epoch_mae_decay),
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Renew thresholds
        thresholds = criterion.dynamic_threshold(thresholds)
        criterion.clear_positive_logits()
        # Write the losses to tensorboard
        write_loss(epoch, 'teaching_source', loss_train, loss_source_dict)
        write_loss(epoch, 'teaching_target', loss_train, loss_target_dict)
        lr_scheduler.step()
        # Selective Retraining
        if (epoch + 1) % args.epoch_retrain == 0:
            model_stu = selective_reinitialize(model_stu, reinit_ckpt, args.keep_modules)
        # Evaluate teacher and student model
        ap50_per_class_teacher, loss_val_teacher = evaluate(
            model=model_tch,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        ap50_per_class_student, loss_val_student = evaluate(
            model=model_stu,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50_tch = np.asarray([ap for ap in ap50_per_class_teacher if ap > -0.001]).mean().tolist()
        map50_stu = np.asarray([ap for ap in ap50_per_class_student if ap > -0.001]).mean().tolist()
        write_ap50(epoch, 'teaching_teacher', map50_tch, ap50_per_class_teacher, idx_to_class)
        write_ap50(epoch, 'teaching_student', map50_stu, ap50_per_class_student, idx_to_class)
        if max(map50_tch, map50_stu) > ap50_best:
            ap50_best = max(map50_tch, map50_stu)
            # CHANGED: Save comprehensive checkpoint including both models
            best_model = model_tch if map50_tch > map50_stu else model_stu
            best_checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': (best_model.module if args.distributed else best_model).state_dict(),
                'teacher_state_dict': (model_tch.module if args.distributed else model_tch).state_dict(),
                'student_state_dict': (model_stu.module if args.distributed else model_stu).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'thresholds': thresholds,
                'map50_teacher': map50_tch,
                'map50_student': map50_stu,
                'args': args
            }
            torch.save(best_checkpoint_dict, output_dir/'model_best.pth')
        if epoch == args.epoch - 1:
            # CHANGED: Save both teacher and student
            last_tch_checkpoint = {
                'epoch': epoch,
                'model_state_dict': (model_tch.module if args.distributed else model_tch).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'thresholds': thresholds,
                'args': args
            }
            torch.save(last_tch_checkpoint, output_dir/'model_last_tch.pth')
            
            last_stu_checkpoint = {
                'epoch': epoch,
                'model_state_dict': (model_stu.module if args.distributed else model_stu).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'thresholds': thresholds,
                'args': args
            }
            torch.save(last_stu_checkpoint, output_dir/'model_last_stu.pth')
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching finished. Time cost: ' + total_time_str + ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


# Evaluate only
def eval_only(model, device):
    if args.distributed:
        Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    # Eval source or target dataset
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    ap50_per_class, epoch_loss_val, coco_data = evaluate(
        model=model,
        criterion=criterion,
        data_loader_val=val_loader,
        output_result_labels=True,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )
    print('Evaluation finished. mAPs: ' + str(ap50_per_class) + '. Evaluation loss: ' + str(epoch_loss_val))
    output_file = output_dir/'evaluation_result_labels.json'
    print("Writing evaluation result labels to " + str(output_file))
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(coco_data, fp)


def main():
    # Initialize distributed mode
    init_distributed_mode(args)
    # Set random seed
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_random_seed(args.random_seed + get_rank())
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    for key, value in args.__dict__.items():
        print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)
    
    # CHANGED: Modified resume logic to distinguish between --resume and --continue-from-checkpoint
    # --resume is used for loading pretrained weights only
    # --continue-from-checkpoint is used for continuing training with optimizer state
    if args.resume != "" and not args.continue_from_checkpoint:
        model = resume_and_load(model, args.resume, device)
    
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    if args.mode == "single_domain":
        single_domain_training(model, device)
    elif args.mode == "cross_domain_mae":
        cross_domain_mae(model, device)
    elif args.mode == "teaching":
        teaching(model, device)
    elif args.mode == "eval":
        eval_only(model, device)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    # Set output directory
    output_dir = Path(args.output_dir)
    logs_dir = output_dir/'data_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # Call main function
    main()
# python main.py --mode single_domain --num_classes 2 --dropout 0.1 --source_dataset tanks_source --target_dataset tanks_test --epoch 50
# python main.py --mode cross_domain_mae --num_classes 2 --dropout 0.0 --source_dataset tanks_source --target_dataset --epoch 50
# python main.py --mode teaching --num_classes 2 --dropout 0.0 --source_dataset tanks_source --target_dataset 
# python main.py --mode eval --num_classes 2 --source_dataset tanks_source --target_dataset

# NEW USAGE EXAMPLES:
# Continue training from a checkpoint (loads model, optimizer, lr_scheduler states):
# python main.py --mode single_domain --continue-from-checkpoint ./output/model_for_eval.pth --epoch 100
# 
# Start training from a specific epoch with a checkpoint:
# python main.py --mode single_domain --continue-from-checkpoint ./output/model_best.pth --start-epoch 50 --epoch 100
