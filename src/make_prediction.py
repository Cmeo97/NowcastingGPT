from einops import rearrange, repeat
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
from pysteps.verification.spatialscores import intensity_scale
from pysteps.visualization import plot_precip_field
import torch.nn.functional as F
from transformers.generation.utils import top_k_top_p_filtering
from typing import List, Union
from torch.distributions.categorical import Categorical




class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device]) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens  
        obs_tokens=rearrange(obs_tokens,'B T H -> B (T H)')
        self.obs_tokens = obs_tokens

        return obs_tokens


    @torch.no_grad()
    def step(self , observations, num_steps) -> None:
        sample=observations
        cond_len = observations.shape[1]
        past = None
        x=sample
        
        

        for k in range(num_steps):
            outputs_wm = self.world_model.forward_with_past(x, past, past_length = (k+cond_len-1))


            if past is None:
                past = [outputs_wm.past_keys_values]
            else:
                past.append(outputs_wm.past_keys_values)
                

            logits = outputs_wm.logits_observations
            logits=logits[:, -1, :]
            logits = top_k_top_p_filtering(logits, top_k=100, top_p=0.95)
            token = Categorical(logits=logits).sample()
            x = token.unsqueeze(1) 
            sample = torch.cat((sample, x), dim=1)
        

        sample = sample[:, :] 


        return sample 

    
    @torch.no_grad()
    def decode_obs_tokens(self, obs_tokens) -> List[Image.Image]:
        generated_sequence=obs_tokens
        generated_sequence=generated_sequence.squeeze(0)
        embedded_tokens = self.tokenizer.embedding(generated_sequence)    
        z = rearrange(embedded_tokens, '(b h w) e -> b e h w', e=1024, h=8, w=8).contiguous()
        rec = self.tokenizer.decode(z, should_postprocess=True)         
        rec= rec.unsqueeze(0)
        return rec



@torch.no_grad()
def generate(batch, tokenizer, world_model, latent_dim, horizon, obs_time):
        
    initial_observations = batch
    device = initial_observations.device
    wm_env = WorldModelEnv(tokenizer, world_model, device)
    input_image = initial_observations[:,:obs_time,:,:,:].to(device=device)
    obs_tokens = wm_env.reset_from_initial_observations(input_image)
    generated_sequence  = wm_env.step(obs_tokens, num_steps=horizon*latent_dim)
    reconstructed_predicted_sequence= wm_env.decode_obs_tokens(generated_sequence)

    return reconstructed_predicted_sequence



def compute_metrics_pre (batch, tokenizer, world_model):
    input_images = batch
    predicted_observations= generate(input_images, tokenizer= tokenizer, world_model= world_model, latent_dim=64, horizon=6, obs_time=3)
    lead_times=6
    print("input_images",input_images.size())
    print("predicted_observations",predicted_observations.size())                                 
    
    avg_metrics = {
        'MSE:': 0, 'MAE:': 0, 'PCC:': 0, 'CSI(1mm):': 0, 'CSI(2mm):': 0, 
        'CSI(8mm):': 0, 'FAR(1mm):': 0, 'FAR(2mm):': 0, 'FAR(8mm):': 0, 
        'FSS(1km):': 0, 'FSS(10km):': 0, 'FSS(20km):': 0, 'FSS(30km):': 0
    }
    
    for i in range(lead_times):
        input_images_npy = input_images[0,i+3,0,:,:].cpu().numpy()*40
        prediction_npy = predicted_observations[0,i+3,0,:,:].cpu().numpy()*40
        scores_cat1 = det_cat_fct(prediction_npy, input_images_npy, 1)
        scores_cat2 = det_cat_fct(prediction_npy, input_images_npy, 2)
        scores_cat8 = det_cat_fct(prediction_npy, input_images_npy, 8)
        scores_cont = det_cont_fct(prediction_npy, input_images_npy, scores = ["MSE", "MAE", "corr_p"], thr=0.1)
        scores_spatial = intensity_scale(prediction_npy, input_images_npy, 'FSS', 0.1, [1,10,20,30])
        
        metrics = {'MSE:': scores_cont['MSE'],
                   'MAE:': scores_cont['MAE'], 
                   'PCC:': scores_cont['corr_p'], 
                   'CSI(1mm):': scores_cat1['CSI'],
                   'CSI(2mm):': scores_cat2['CSI'],
                   'CSI(8mm):': scores_cat8['CSI'],
                   'FAR(1mm):': scores_cat1['FAR'],
                   'FAR(2mm):': scores_cat2['FAR'],
                   'FAR(8mm):': scores_cat8['FAR'],
                   'FSS(1km):': scores_spatial[3][0],
                   'FSS(10km):': scores_spatial[2][0],
                   'FSS(20km):': scores_spatial[1][0],
                   'FSS(30km):': scores_spatial[0][0]
        }
        
        # Update avg_metrics dictionary
        for key in avg_metrics:
            avg_metrics[key] += metrics[key]
        
    # Compute average for each metric
    for key in avg_metrics:
        avg_metrics[key] = np.around(avg_metrics[key] / lead_times, 3)
    
    return avg_metrics



def make_predictions_from_batch(batch, save_dir, epoch, tokenizer, world_model):
    input_images = batch
    predicted_observations= generate(input_images, tokenizer= tokenizer, world_model= world_model, latent_dim=64, horizon=6, obs_time=3)
    lead_times=6
    print("input_images",input_images.size())
    print("predicted_observations",predicted_observations.size()) 

    for i in range(lead_times):
        input_images_npy = input_images[0,i+3,0,:,:].cpu().numpy()*40
        prediction_npy = predicted_observations[0,i+3,0,:,:].cpu().numpy()*40

        # Plot the precipitation fields using your plot_precip_field function
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_precip_field(input_images_npy, title="Input")
        
        plt.subplot(1, 2, 2)
        plot_precip_field(prediction_npy, title="Prediction")

        plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_t_{i:03d}.png'))

        # Optionally, display the figure if needed
        plt.show()

        # Close the figure to free up resources
        plt.close()

    return




    

