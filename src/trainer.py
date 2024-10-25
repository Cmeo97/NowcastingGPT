from collections import defaultdict
from pathlib import Path
import shutil
import sys
import time
import math
from typing import Any, Dict
import lightning_fabric as L
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import wandb
from collector import Collector
from make_reconstructions import make_reconstructions_from_batch, generate_reconstructions_with_tokenizer,compute_metrics
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import configure_optimizer, set_seed, extract_state_dict
from make_prediction import compute_metrics_pre, make_predictions_from_batch
from pycm import *

class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        #self.device = torch.device(cfg.common.device)
        self.fabric = L.Fabric(accelerator=cfg.common.device, devices=1, precision="16-mixed")
        self.fabric.launch()
        self.batch_size_training = cfg.common.batch_size_training
        self.batch_size_testing = cfg.common.batch_size_testing
        self.obs_time = cfg.common.obs_time 
        self.pred_time = cfg.common.pred_time 
        self.time_interval = cfg.common.time_interval
        self.optimizer_filename = cfg.checkpoint_OPT.name_to_checkpoint
        self.ckpt_dir = Path('/space/ankushroy/Checkpoint_Iris_classifier_exp_5')
        self.media_dir = Path('media')
        self.reconstructions_dir = self.media_dir / 'reconstructions'
        self.prediction_dir= self.media_dir / 'predictions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            self.ckpt_dir.mkdir(exist_ok=True, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)
            self.prediction_dir.mkdir(exist_ok=False, parents=False)
        
        #################################################################

        if self.cfg.training.should:
            self.train_collector = Collector()


        if self.cfg.evaluation.should:
            self.test_collector = Collector()
            
        assert self.cfg.training.should or self.cfg.evaluation.should

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, config=instantiate(cfg.world_model))
        
        #self.classifier = world_model.head_classification.to(self.device)
        self.classifier = world_model.head_classification.to(self.fabric.device)
        #self.agent = Agent(tokenizer, world_model).to(self.device)
        #self.agent = Agent(tokenizer, world_model).to(self.fabric.device)
        self.tokenizer = tokenizer.to(self.fabric.device)
        self.world_model = world_model.to(self.fabric.device)

        print(f'{sum(p.numel() for p in self.tokenizer.parameters())} parameters in tokenizer')
        print(f'{sum(p.numel() for p in self.world_model.parameters())} parameters in world_model')
        print(f'{sum(p.numel() for p in self.classifier.parameters())} parameters in classifier')

        self.optimizer_tokenizer = torch.optim.Adam(self.tokenizer.parameters(), lr=cfg.training.learning_rate)
        self.optimizer_world_model = configure_optimizer(self.world_model, cfg.training.learning_rate, cfg.training.world_model.weight_decay)
        self.optimizer_classifier = configure_optimizer(self.classifier, cfg.training.learning_rate, cfg.training.world_model.weight_decay)

        if cfg.initialization.path_to_checkpoint is not None:
            #self.agent.load(**cfg.initialization, device=self.device)
            self.load(**cfg.initialization, device=self.fabric.device)
        if cfg.checkpoint_OPT.load_opti:
            self.load_checkpoint()

        

    def run(self) -> None:
               
        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            if self.cfg.training.should:
                training_data_dataloader, length_train = self.train_collector.collect_training_data(self.cfg.common.path_to_training_data, self.batch_size_training)
                training_data_dataloader = self.fabric.setup_dataloaders(training_data_dataloader)
                if epoch <= self.cfg.collection.train.stop_after_epochs:

                    to_log += self.train_agent(epoch, training_data_dataloader)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                testing_data_dataloader, length_test= self.test_collector.collect_testing_data_ext(self.cfg.common.path_to_testing_data_ext, self.batch_size_testing)
                testing_data_dataloader = self.fabric.setup_dataloaders(testing_data_dataloader)
                
                to_log += self.eval_agent(epoch, testing_data_dataloader, length_test)
                

            if self.cfg.training.should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})

        self.finish()

    def train_agent(self, epoch: int, training_data_dataloader) -> None:
        self.tokenizer.train() or self.world_model.train()
        self.tokenizer.zero_grad() or self.world_model.zero_grad()
        

        metrics_tokenizer, metrics_world_model, metrics_classifier = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_classifier = self.cfg.training.classifier


        if epoch >= cfg_tokenizer.start_after_epochs and epoch <= cfg_tokenizer.stop_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for batch in training_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_tokenizer, loss, intermediate_los = self.train_component(self.tokenizer, self.optimizer_tokenizer,batch, loss_total_epoch, intermediate_losses, train_world_model=False, **cfg_tokenizer) 
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
            print("tokenizer_loss_total_epoch", loss_total_epoch)
        self.tokenizer.eval()


        if epoch >= cfg_classifier.start_after_epochs and epoch <= cfg_classifier.stop_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)                    
            for batch in training_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_classifier, loss, intermediate_los = self.train_component(self.world_model, self.optimizer_classifier, batch, loss_total_epoch, intermediate_losses, tokenizer=self.tokenizer, train_world_model=False, **cfg_classifier)
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
                print("classifier_total_loss", loss_total_epoch)
        self.world_model.eval()

        if epoch >= cfg_world_model.start_after_epochs and epoch <= cfg_world_model.stop_after_epochs:
            loss_total_epoch = 0.0
            intermediate_losses = defaultdict(float)
            for batch in training_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_world_model, loss, intermediate_los = self.train_component(self.world_model, self.optimizer_world_model, batch, loss_total_epoch, intermediate_losses, tokenizer=self.tokenizer, train_world_model=True, **cfg_world_model)
                loss_total_epoch = loss 
                intermediate_losses = intermediate_los
            print("worldmodel_loss_total_epoch", loss_total_epoch)
        self.world_model.eval()
        
        return [{'epoch': epoch, **metrics_tokenizer, **metrics_world_model, **metrics_classifier}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, batch,  loss_total_epoch, intermediate_losses,  batch_num_samples: int, grad_acc_steps: int, train_world_model: bool, **kwargs_loss: Any) -> Dict[str, float]:
        mini_batch= math.floor(batch.size(0)/(batch_num_samples*grad_acc_steps))
        counter=0

        for _ in range(mini_batch):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps): 
                batch_training= batch[(counter*batch_num_samples):(counter+1)*(batch_num_samples),:,:,:,:] 
                batch_training = batch_training.to(self.fabric.device)
                losses = component.compute_loss(batch_training, train_world_model=train_world_model, **kwargs_loss) / grad_acc_steps
                loss_total_step = losses.loss_total
                self.fabric.backward(loss_total_step)
                loss_total_epoch += loss_total_step.item() 

                for loss_name, loss_value in losses.intermediate_losses.items():
                    intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value
                
                counter= counter + 1
                
            optimizer.step()

            
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}


        return metrics, loss_total_epoch, intermediate_losses
    
    
    @torch.no_grad()
    def eval_agent(self, epoch: int, testing_data_dataloader, length_test) -> None:
        self.tokenizer.eval() or self.world_model.eval()
        metrics_tokenizer, metrics_world_model, metrics_classifier = {}, {}, {}

        all_true_labels = []
        all_pred_labels = []
        

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_classifier = self.cfg.evaluation.classifier
        
        # TOKENISER: 
        if epoch >= cfg_tokenizer.start_after_epochs and epoch <= cfg_tokenizer.stop_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)
            self.accumulated_metrics = defaultdict(float)           
            for batch in testing_data_dataloader:
                batch= batch.unsqueeze(2)            
                metrics_tokenizer, loss_test, intermediate_los  = self.eval_component(self.tokenizer, batch, loss_total_test_epoch, intermediate_losses, train_world_model=False)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss tokeniser", loss_total_test_epoch)
                
            for metrics_name, metrics_value in metrics_tokenizer.items():
                metrics_tokenizer[metrics_name] = metrics_value / length_test

        # CLASSIFIER ONLY:
        if epoch >= cfg_classifier.start_after_epochs and epoch <= cfg_classifier.stop_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)        
            for batch in testing_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics, loss_test, intermediate_los = self.eval_component(self.world_model, self.tokenizer, batch, loss_total_test_epoch, intermediate_losses, tokenizer=self.tokenizer, train_world_model=False)
                metrics_classifier.update(metrics)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss classifier", loss_total_test_epoch)
                true_labels, pred_labels = self.world_model.evaluation_conf_matrix(batch, self.tokenizer, train_world_model=False)
                all_true_labels.extend(true_labels)
                #print(" true labels", all_true_labels)
                all_pred_labels.extend(pred_labels)
                #print("predicted labels", all_pred_labels)                

            cm = ConfusionMatrix(actual_vector = all_true_labels, predict_vector = all_pred_labels, classes=[0,1])
            
            if cm.F1_Macro is None or cm.F1_Macro == 'None':
                f1_score = 0.0
            else:
                f1_score = cm.F1_Macro

            if cm.PPV_Macro is None or cm.PPV_Macro == 'None':
                precision = 0.0
            else:
                precision = cm.PPV_Macro
            
            if cm.TPR_Macro is None or cm.TPR_Macro == 'None':
                recall = 0.0
            else:
                recall = cm.TPR_Macro


            # Append the F1 score to the metrics_classifier dictionary
            metrics_classifier["F1_score"] = f1_score
            metrics_classifier["Precsion_score"] = precision 
            metrics_classifier["Recall Score"] = recall
               
            
        # WORLD MODEL:
        if epoch >= cfg_world_model.start_after_epochs and epoch <= cfg_world_model.stop_after_epochs:
            loss_total_test_epoch = 0.0
            intermediate_losses = defaultdict(float)
            self.accumulated_metrics = defaultdict(float)         
            for batch in testing_data_dataloader:
                batch= batch.unsqueeze(2)
                metrics_world_model, loss_test, intermediate_los = self.eval_component(self.world_model, self.tokenizer, batch, loss_total_test_epoch, intermediate_losses, tokenizer=self.tokenizer, train_world_model=True)
                loss_total_test_epoch = loss_test 
                intermediate_losses = intermediate_los
                print("evaluation total loss world model", loss_total_test_epoch)

            for metrics_name, metrics_value in metrics_world_model.items():
                metrics_world_model[metrics_name] = metrics_value / length_test


        if cfg_tokenizer.save_reconstructions:
            for batch in testing_data_dataloader:
                reconstruct_batch= batch.unsqueeze(2)
                break
            reconstruct_batch = reconstruct_batch.to(self.fabric.device)
            make_reconstructions_from_batch(reconstruct_batch, save_dir=self.reconstructions_dir, epoch=epoch, tokenizer=self.tokenizer)
        
        if cfg_world_model.save_predictions:
            for batch in testing_data_dataloader:
                prediction_batch= batch.unsqueeze(2)
                break
            prediction_batch = prediction_batch.to(self.fabric.device)
            make_predictions_from_batch(prediction_batch, save_dir=self.prediction_dir, epoch=epoch, tokenizer=self.tokenizer, world_model=self.world_model)



        return [metrics_tokenizer, metrics_world_model, metrics_classifier]
    
    

    @torch.no_grad()
    def eval_component(self, component: nn.Module, component1: nn.Module, batch, loss_total_test_epoch, intermediate_losses, train_world_model: bool, **kwargs_loss: Any) -> Dict[str, float]:
        pysteps_metrics = {}
        
        batch_testing = batch
        batch_testing = batch_testing.to(self.fabric.device)        
        losses = component.compute_loss(batch_testing, **kwargs_loss, train_world_model=train_world_model)
        loss_total_test_epoch += (losses.loss_total.item())

        for loss_name, loss_value in losses.intermediate_losses.items():
            intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

        ######## Pysteps Metrics Calculation
        
        if str(component) =='tokenizer':
            rec_frames = generate_reconstructions_with_tokenizer(batch_testing, component)
            pysteps_metrics = compute_metrics(batch_testing, rec_frames)
        
            for metrics_name, metrics_value in pysteps_metrics.items():
                if math.isnan(metrics_value):
                    metrics_value = 0.0
                self.accumulated_metrics[metrics_name] += metrics_value
        

            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {**intermediate_losses, **self.accumulated_metrics}
        
        elif train_world_model == True:

            pysteps_metrics = compute_metrics_pre(batch_testing, component1, component)
        
            for metrics_name, metrics_value in pysteps_metrics.items():
                if math.isnan(metrics_value):
                    metrics_value = 0.0
                self.accumulated_metrics[metrics_name] += metrics_value
        

            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {**intermediate_losses, **self.accumulated_metrics}
        
        else: 
            intermediate_losses = {k: v  for k, v in intermediate_losses.items()}
            metrics = {**intermediate_losses}
        
        return metrics, loss_total_test_epoch, intermediate_losses
    

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        if epoch % self.cfg.evaluation.every == 0:
            state_to_save = {
                'world_model': self.world_model.state_dict(),   
            }
            torch.save(state_to_save, self.ckpt_dir / f'model_checkpoint_epoch_{epoch:02d}.pt')
            if not save_agent_only:
                torch.save({
                    "optimizer_world_model": self.optimizer_world_model.state_dict(),  
                }, self.ckpt_dir / f'optimizer_{epoch:02d}.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        self._save_checkpoint(epoch, save_agent_only)


    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        ckpt_opt = torch.load(self.ckpt_dir / self.optimizer_filename, map_location=self.fabric.device)
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        print(f'Successfully loaded optimizer from {self.ckpt_dir.absolute()}.')


    def load(self, path_to_checkpoint: Path, path_to_checkpoint_trans: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True) -> None: 
        agent_state_dict = torch.load(path_to_checkpoint, device)
        if path_to_checkpoint_trans is not None:
            agent_state_dict_trans = torch.load(path_to_checkpoint_trans, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
            print("Tokenizer checkpoint uploaded successfully")
        if load_world_model:
            world_model_state_dict = agent_state_dict_trans['world_model']
            self.world_model.load_state_dict(world_model_state_dict)
            print("World Model checkpoint uploaded successfully")    
    
    def _to_device(self, batch: torch.Tensor):
        return batch.to(self.device)

    def _out_device(self, batch: torch.Tensor):
        return batch.detach()
    
    def finish(self) -> None:
        wandb.finish()

    
