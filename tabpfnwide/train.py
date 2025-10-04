# Partly taken from TabICL run.py script https://github.com/soda-inria/tabicl/blob/main/src/tabicl/train/run.py
from contextlib import nullcontext
import datetime
import os
from tabpfn.model.loading import load_model_criterion_config
from tabpfn.model.config import ModelConfig
from tabpfn.model.memory import MemoryUsageEstimator
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np
from torch import nn
from tabicl.prior.dataset import PriorDataset
from tabicl.train.optim import get_cosine_with_restarts
from tabicl.train.run import Timer
from training_parser import parse_args
from data import get_wide_validation_data, load_prior_dataloader, PriorDataLoaderConfig, PriorDatasetConfig, get_validation_datasets
from utils import PredictionResults, get_new_features
import wandb
import tqdm
from dataclasses import dataclass, asdict, fields
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", module="sklearn")


@dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_steps: int = 10000
    use_wandb: bool = False
    d_type: str = "float16"  # Options: "float16", "float32"
    warmup_proportion: int = 0.02
    num_cycles: int = 10
    gradient_clipping: float = 1.0
    validation_interval: int = 200
    validation_interval_wide: int = 200
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100
    resume_checkpoint: str = None
    use_original_model: bool = False
    model_path: str = None
    validation_datasets = ["COAD", "gbm"]
    omic_combinations = [["mrna"], ["methylation"], ["mrna", "methylation", "mirna"]]
    
    
@dataclass
class FeatureAddingConfig:
    add_features_min: int = 0
    add_features_max: int = 0
    warmup_steps: int = 0
    min_sparsity: float = 0.0
    max_sparsity: float = 0.2
    min_noise: float = 0.0
    max_noise: float = 0.1


class Trainer:
    train_config: TrainConfig = TrainConfig()
    def __init__(self, parsed_args=defaultdict(dict)):
        self.train_config = TrainConfig(**parsed_args["train_config"])
        self.batch_size = self.train_config.batch_size
        self.model_config = ModelConfig(**parsed_args["model_config"])
        self.feature_adding_config = FeatureAddingConfig(**parsed_args["feature_adding_config"])
        self.criterion = nn.CrossEntropyLoss()        
        self.configure_ddp()
        self.prior_dataset_config = PriorDatasetConfig(batch_size=self.batch_size, **parsed_args["prior_dataset_config"])
        self.prior_dataloader_config = PriorDataLoaderConfig(pin_memory_device=self.device, **parsed_args["prior_dataloader_config"])
        self.load_model()
        self.configure_amp()
        self.start_time = datetime.datetime.now()
        self.curr_step = 0
        self.dataloader = iter(load_prior_dataloader(PriorDataset, self.prior_dataset_config, self.prior_dataloader_config))
        if self.train_config.resume_checkpoint:
            self.load_checkpoint()
        self.configure_wandb()

    
    def configure_ddp(self):
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=1800),
            )
            self.device = f"cuda:{rank}"
            torch.cuda.set_device(self.device)
            self.is_main_process = rank == 0
            self.batch_size = self.batch_size // world_size
        
        seed_offset = rank if self.ddp else 0
        np.random.seed(42 + seed_offset)
        torch.manual_seed(44 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    
    def configure_wandb(self):
        if self.train_config.use_wandb and self.is_main_process:
            self.wandb_obj = wandb.init(
                project="tabpfn", entity="modexta", 
                config= asdict(self.model_config) | asdict(self.prior_dataset_config) | asdict(self.train_config),
                resume="allow",
                id=self.wandb_id if self.train_config.resume_checkpoint else None,
            )
            
    def save_checkpoint(self, name):
        os.makedirs(self.train_config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.train_config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.model.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "step": self.curr_step,
            "wandb_id": self.wandb_obj.id if self.train_config.use_wandb else None,
        }
        torch.save(checkpoint, checkpoint_path)
        
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.train_config.resume_checkpoint, map_location=self.device, weights_only=False)
        self.model.module.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.curr_step = checkpoint["step"]
        self.wandb_id = checkpoint.get("wandb_id", None)
        
            
    def load_model(self):
        if self.train_config.use_original_model:
            model, _, config = load_model_criterion_config(
                model_path=None,
                check_bar_distribution_criterion=False,
                cache_trainset_representation=False,
                which='classifier',
                version='v2',
                download=True,
            )
            # Disable feature grouping
            model.features_per_group = 1
            config.features_per_group = 1
            # Compare loaded config to self.model_config and assert all fields are equal
            for field in fields(self.model_config):
                loaded_value = getattr(config, field.name, None)
                current_value = getattr(self.model_config, field.name, None)
                assert loaded_value == current_value, f"Config mismatch in field '{field.name}': loaded={loaded_value}, expected={current_value}"
        else:
            raise NotImplementedError("Loading untrained model deprecated.")
    
        if self.ddp:
            model = model.to(self.device)
            model_ = DDP(model, device_ids=[int(self.device.split(':')[-1])], broadcast_buffers=False)
        else:
            model_ = model.to(self.device)

        self.model = model_
        self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=self.train_config.learning_rate, weight_decay=self.train_config.weight_decay)
        
        self.scheduler = get_cosine_with_restarts(
            self.optimizer, int(self.train_config.num_steps * self.train_config.warmup_proportion), 
            self.train_config.num_steps, self.train_config.num_cycles
        )
        
        
    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = "cuda" in self.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            self.amp_ctx = torch.autocast(
                device_type="cuda", dtype=torch.float16 if self.train_config.d_type == "float16" else torch.float32
            )
        else:
            self.amp_ctx = nullcontext()
            
            
    def get_feature_adding_parameters(self):
        if self.feature_adding_config.warmup_steps > 0 and self.curr_step < self.feature_adding_config.warmup_steps:
            max_features_add = self.feature_adding_config.add_features_min + \
            (self.feature_adding_config.add_features_max - self.feature_adding_config.add_features_min) * \
            (self.curr_step / self.feature_adding_config.warmup_steps)
        else:
            max_features_add = self.feature_adding_config.add_features_max
        new_features = np.random.randint(self.feature_adding_config.add_features_min, max_features_add + 1)
        sparsity = np.random.uniform(self.feature_adding_config.min_sparsity, self.feature_adding_config.max_sparsity)
        noise = np.random.uniform(self.feature_adding_config.min_noise, self.feature_adding_config.max_noise)
        return new_features, sparsity, noise
    
        
    def validate(self):
        self.model.eval()
        pred_res = []
        val_losses = []
        for dataset in get_wide_validation_data(self.device, self.train_config.validation_datasets, self.train_config.omic_combinations):
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = dataset

            
            MemoryUsageEstimator.reset_peak_memory_if_required(
                save_peak_mem=True,
                model=self.model.module,
                X=torch.cat([X_train_tensor, X_test_tensor], dim=0),
                cache_kv=False,
                dtype_byte_size= 2 if self.train_config.d_type == "float16" else 4,
                device=torch.device(self.device),
                safety_factor=1.2, 
            )
            
            with torch.inference_mode():
                with self.amp_ctx:
                    pred_logits = self.model(
                        train_x=X_train_tensor,
                        train_y=y_train_tensor,
                        test_x=X_test_tensor,
                    )
                    n_classes = len(np.unique(y_train_tensor.cpu()))
                    pred_logits = pred_logits[..., :n_classes].float()  
                    pred_probs = torch.softmax(pred_logits, dim=-1)[:, 0, :].detach().cpu().numpy()
                val_loss = self.criterion(pred_logits.reshape(-1, n_classes), y_test_tensor.flatten().long())
                val_losses.append(val_loss.item())
            pred_res.append(PredictionResults(y_test_tensor.flatten().cpu().numpy(), pred_probs))
        
        self.model.module.reset_save_peak_mem_factor(None)
        
        if self.train_config.use_wandb:
            mean_val_loss = np.mean(val_losses)
            mean_val_accuracy = np.mean([res.get_classification_report(print_report=False)['accuracy'] for res in pred_res])
            mean_val_f1_weighted = np.mean([res.get_f1_score(average='weighted') for res in pred_res])
            rocs = []
            for res in pred_res:
                try:
                    rocs.append(res.get_roc_auc_score(multi_class='ovo'))
                except ValueError:
                    rocs.append(np.nan)
            mean_val_roc_auc = np.nanmean(rocs)
            
            wandb.log({
                f"validation_loss_wide": mean_val_loss,
                f"validation_accuracy_wide": mean_val_accuracy,
                f"validation_f1_weighted_wide": mean_val_f1_weighted,
                f"validation_roc_auc_wide": mean_val_roc_auc,
                "custom_step": self.curr_step,
            })
        
        
    def train(self):
        oom_errors = 0
        
        step_progress = (tqdm.tqdm(range(self.curr_step, self.train_config.num_steps)) 
            if self.is_main_process 
            else range(self.curr_step, self.train_config.num_steps)
        )
        
        for i in step_progress:
            self.curr_step = i
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with Timer() as timer:
                batch = next(self.dataloader)
            prior_time = timer.elapsed
            
            X, y, d, seq_len, trainsizes = batch
            if not (torch.all(d == d[0]) and torch.all(seq_len == seq_len[0]) and torch.all(trainsizes == trainsizes[0])):
                continue
            X = X[:, :, :d[0]]
            if self.feature_adding_config.add_features_max > 0:
                new_features, sparsity, noise = self.get_feature_adding_parameters()
                X = get_new_features(X, new_features, sparsity=sparsity, noise_std=noise)
                
                
            memory_remainder = MemoryUsageEstimator.estimate_memory_remainder_after_batch(
                X,
                self.model.module,
                cache_kv=False,
                device=torch.device(self.device),
                dtype_byte_size=2 if self.train_config.d_type == "float16" else 4,
                safety_factor=4,
            )
            should_skip_tensor = torch.tensor(int(memory_remainder < 0), device=self.device)
            torch.distributed.all_reduce(should_skip_tensor, op=torch.distributed.ReduceOp.SUM)
            # If any rank wants to skip, all should skip
            if should_skip_tensor.item() > 0:
                print(f"Skipping step {self.curr_step} due to memory constraints.")
                continue
                    
            X_train = X[:, :trainsizes[0]].transpose(0, 1).to(self.device)
            X_test = X[:, trainsizes[0]:].transpose(0, 1).to(self.device)
            y_train = y[:, :trainsizes[0]].transpose(0, 1).to(self.device)
            y_test = y[:, trainsizes[0]:].transpose(0, 1).to(self.device)

            try:
                with Timer() as timer:
                    with self.amp_ctx:
                        pred_logits = self.model(
                            train_x=X_train,
                            train_y=y_train,
                            test_x=X_test,
                        )
                        pred_logits = pred_logits.float()
                    loss = self.criterion(pred_logits.reshape(-1, 10), y_test.flatten().long())
                    self.scaler.scale(loss).backward()
                forward_time = timer.elapsed
            except torch.cuda.OutOfMemoryError:
                oom_errors += 1
                torch.cuda.empty_cache()
                if oom_errors / self.curr_step > 0.1:
                    raise RuntimeError("Too many OOM errors, stopping training.")
                continue
            torch.cuda.empty_cache()
            
            if self.train_config.gradient_clipping > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.gradient_clipping)

            # Update parameters
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update the learning rate
            self.scheduler.step()

            if self.is_main_process and self.train_config.use_wandb:
                wandb.log({
                    "loss": loss.item(), 
                    "lr": self.optimizer.param_groups[0]['lr'], 
                    "oom_errors": oom_errors, 
                    "prior_time": prior_time,
                    "forward_time": forward_time,
                    "total_datasets" : self.curr_step * self.train_config.batch_size,
                    "max_features_added": new_features if self.feature_adding_config.add_features_max > 0 else 0,
                    "custom_step": self.curr_step,
                })
                
                
            if self.is_main_process and self.train_config.validation_interval_wide > 0 and self.curr_step % self.train_config.validation_interval_wide == 0:
                print("Validating wide...")
                self.validate()
                
            if self.is_main_process and self.curr_step % self.train_config.save_interval == 0:
                self.save_checkpoint(f"{self.start_time.strftime('%Y%m%d_%H%M%S')}_step_{self.curr_step}_{self.wandb_obj.name if self.train_config.use_wandb else 'no_wandb'}.pt")
                
        if self.ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    args_dict = parse_args()
    try:
        trainer = Trainer(args_dict)
        trainer.train()
    finally:
        dist.destroy_process_group()
    
    