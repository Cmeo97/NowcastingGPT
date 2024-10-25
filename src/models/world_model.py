from dataclasses import dataclass
from typing import Any, Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses, Batch
from .transformer_classifier import ViT


@dataclass
class WorldModelOutput_classifier:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_classification: torch.FloatTensor
    past_keys_values: torch.tensor

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    past_keys_values: torch.tensor    

class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size = obs_vocab_size
        self.config = config
        self.transformer = Transformer(config)
        obs_tokens_pattern = torch.ones(config.tokens_per_block)
        

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        
        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        
        self.head_classification = ViT(num_frames=config.max_blocks,
                                    tokens_per_frame=config.tokens_per_block,embed_dim=128, dim=128, 
                                    depth=6,heads=8,mlp_dim=256, num_classes=2, pool='cls',
                                        dropout=0., emb_dropout=0., dim_head=16)
        
        
        
        self.apply(init_weights)
           

    def __repr__(self) -> str:
        return "world_model"
                    
                
    
    def forward(self, obs_tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput_classifier:

        num_steps = obs_tokens.shape[1]  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        sequences = self.embedder(obs_tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=obs_tokens.device))

        x = self.transformer.forward(sequences, past_keys_values)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_classification = self.head_classification(x)
        
        return WorldModelOutput_classifier(x, logits_observations, logits_classification, past_keys_values)

    
    def forward_with_past(self, obs_tokens: torch.LongTensor, past_keys_values=None, past_length = None) -> WorldModelOutput:
        # inference only
        assert not self.training
        num_steps = obs_tokens.shape[1]  # (B, T)
        
        if past_keys_values is not None:
            assert past_length is not None
            past_keys_values= torch.cat(past_keys_values, dim=-2) 
            past_shape = list(past_keys_values.shape)
            expected_shape = [self.config.num_layers, 2, obs_tokens.shape[0], self.config.num_heads, past_length, self.config.embed_dim//self.config.num_heads]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
        else:
            past_length = 0
        a = self.embedder(obs_tokens, num_steps, past_length)
        
        b =  self.pos_emb(past_length + torch.arange(num_steps, device=obs_tokens.device))
        sequences = a + b 
        x, past_keys_values = self.transformer.forward_with_past(sequences, past_keys_values)
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=past_length)

        return WorldModelOutput(x, logits_observations, past_keys_values)
    
    
    
    
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, train_world_model: bool, **kwargs: Any) -> LossWithIntermediateLosses:
        
        with torch.no_grad():
            observations= rearrange(batch, 'b t c h w  -> (b t) c h w')
            obs_tokens = tokenizer.encode(observations, should_preprocess=True).tokens  # (BL, K)
            shape_obs = batch.size()
            shape_token= obs_tokens.size()
            

        b = shape_obs[0]
        l = shape_obs[1]
        k = shape_token[1]
        
        tokens = obs_tokens.view(b, l*k)  # (B, L(K))
        outputs = self.forward(tokens)
        
        labels_classification = self.compute_labels_classification(observations).to(tokens.device)
        
        if train_world_model == True:
            
            evt_loss_weight = 1.0
            labels_observations = self.compute_labels_world_model(tokens)
            prob_classification = F.softmax(outputs.logits_classification, dim=1).to(tokens.device)
            labels_classification = labels_classification.unsqueeze(1).to(tokens.device)
            loss_evl = self.cal_evt_loss(prob_classification, labels_classification)

            # World Model loss:
            logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
            loss_obs = F.cross_entropy(logits_observations,labels_observations) + evt_loss_weight * loss_evl
            return LossWithIntermediateLosses(loss_obs = loss_obs)
        else:                                                               
            loss_classification = F.cross_entropy(outputs.logits_classification, labels_classification)
            return LossWithIntermediateLosses(loss_classification = loss_classification)


    def compute_labels_world_model(self, obs_tokens: torch.Tensor):
        labels_observations = obs_tokens[:, 1:] 
        return labels_observations.reshape(-1)


    # Function for calculating the classification labels
    def compute_labels_classification(self, batch: torch.Tensor) -> torch.Tensor:
        batch = rearrange(batch, '(b t) c h w  -> b t c h w', t = 9)
        mean_prec = batch.mean(dim=3).mean(dim=3).squeeze(2)  
        mean_prec = mean_prec.mean(dim=1)
        mean_prec = mean_prec.to(batch.device)
        threshold_e = 0.01        # threshold for extreme event
        threshold = torch.ones(mean_prec.size()) * threshold_e
        threshold  = threshold.to(batch.device)
        labels_classification = (mean_prec > threshold).long() # (b, s)
        labels_classification = labels_classification.to(batch.device)  # (b, s, 1) 
        return labels_classification    
    
    
    # EVL loss function
    def cal_evt_loss(self, indicator, gt_indicator, gamma = 1, beta0 = 0.95, beta1 = 0.05, epsilon = 1e-7):
        loss1 = -1 * beta0 * torch.pow((1-indicator/gamma),gamma) * gt_indicator * torch.log(indicator + epsilon)
        loss2 = -1 * beta1 * torch.pow((1-(1-indicator)/gamma),gamma) * (1-gt_indicator) * torch.log(1-indicator + epsilon)
        loss = loss1 + loss2 
        return loss.mean()
    
    