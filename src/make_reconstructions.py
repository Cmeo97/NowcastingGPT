from einops import rearrange
import numpy as np
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.detcontscores import det_cont_fct
from pysteps.verification.spatialscores import intensity_scale
from pysteps.visualization import plot_precip_field


@torch.no_grad()
def make_reconstructions_from_batch(batch, save_dir, epoch, tokenizer):
    #check_batch(batch)

    original_frames = rearrange(batch, 'b t c h w  -> (b t) c h w')
    batch_tokenizer = batch

    rec_frames = generate_reconstructions_with_tokenizer(batch_tokenizer, tokenizer)

    for i in range(6):
        original_frame = original_frames[i,0,:,:]
        a_display = tensor_to_np_frames(original_frame)
        rec_frame = rec_frames[i,0,:,:]
        ar_display = tensor_to_np_frames(rec_frame)

        # Plot the precipitation fields using your plot_precip_field function
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_precip_field(a_display, title="Input")
        
        plt.subplot(1, 2, 2)
        plot_precip_field(ar_display, title="Reconstruction")

        plt.savefig(os.path.join(save_dir, f'epoch_{epoch:03d}_t_{i:03d}.png'))

        # Optionally, display the figure if needed
        plt.show()

        # Close the figure to free up resources
        plt.close()

    return


def tensor_to_np_frames(inputs):
    return inputs.cpu().numpy()*40 


@torch.no_grad()
def generate_reconstructions_with_tokenizer(batch, tokenizer):
    inputs = rearrange(batch, 'b t c h w  -> (b t) c h w')
    outputs = reconstruct_through_tokenizer(inputs, tokenizer)
    rec_frames = outputs
    return rec_frames


@torch.no_grad()
def reconstruct_through_tokenizer(inputs, tokenizer):
    reconstructions = tokenizer.encode_decode(inputs, should_preprocess=True, should_postprocess=True)
    return torch.clamp(reconstructions, 0, 1)

def compute_metrics (batch, rec_frames):
    input_images = rearrange(batch, 'b t c h w  -> (b t) c h w')
    input_images = input_images.squeeze(1)                                  
    reconstruction = rec_frames.squeeze(1) 

    total_images = input_images.shape[0]
                                     
    avg_metrics = {
        'MSE:': 0, 'MAE:': 0, 'PCC:': 0, 'CSI(1mm):': 0, 'CSI(2mm):': 0, 
        'CSI(8mm):': 0, 'FAR(1mm):': 0, 'FAR(2mm):': 0, 'FAR(8mm):': 0, 
        'FSS(1km):': 0, 'FSS(10km):': 0, 'FSS(20km):': 0, 'FSS(30km):': 0
    }

    
    
    for i in range(total_images):
        input_images_npy = tensor_to_np_frames(input_images[i])
        reconstruction_npy = tensor_to_np_frames(reconstruction[i])
        scores_cat1 = det_cat_fct(reconstruction_npy, input_images_npy, 1)
        scores_cat2 = det_cat_fct(reconstruction_npy, input_images_npy, 2)
        scores_cat8 = det_cat_fct(reconstruction_npy, input_images_npy, 8)
        scores_cont = det_cont_fct(reconstruction_npy, input_images_npy, scores = ["MSE", "MAE", "corr_p"], thr=0.1)
        scores_spatial = intensity_scale(reconstruction_npy, input_images_npy, 'FSS', 0.1, [1,10,20,30])
        
        metrics = {'MSE:': scores_cont['MSE'],
                   'MAE:': scores_cont['MAE'], 
                   'PCC:': scores_cont['corr_p'], 
                   'CSI(1mm):': scores_cat1['CSI'],
                   'CSI(2mm):': scores_cat2['CSI'],
                   'CSI(8mm):': scores_cat8['CSI'],
                   'FAR(1mm):': scores_cat1['FAR'],
                   'FAR(2mm):': scores_cat2['FAR'],
                   'FAR(8mm):': scores_cat8['FAR'],
                   'FSS(1km):': scores_spatial[0][0],
                   'FSS(10km):': scores_spatial[1][0],
                   'FSS(20km):': scores_spatial[2][0],
                   'FSS(30km):': scores_spatial[3][0]
        }
        
        # Update avg_metrics dictionary
        for key in avg_metrics:
            avg_metrics[key] += metrics[key]
        
    # Compute average for each metric
    for key in avg_metrics:
        avg_metrics[key] = np.around(avg_metrics[key] / total_images, 3)
    
    return avg_metrics