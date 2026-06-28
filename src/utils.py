import gc
import cv2
import time
import torch
from src.config import VOCAB
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

BLANK_INDEX = 0
def clear_gpu_memory():
    """Frees cached memory aggressively on a low-VRAM home GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")


#---------------------------------------------------------------------------------------------------

def resize_keep_aspect(img, target_h=64):
    """
    Resizes an incoming image so its height matches the network requirement (64 pixels).
    Crucially scales the width proportionally to prevent the text from squishing or stretching.
    """
    # 1. Extract the current height and width from the image array dimensions
    h, w = img.shape[:2]

    # 2. Calculate the scaling multiplier (target height divided by original height)
    # Then multiply the original width by this ratio to find the proportional new width
    new_w = int(w * (target_h / h))
    
    # 3. SAFETY CHECK: If an image is extremely narrow, the math could round down to 0 width.
    # A width of 0 will crash OpenCV's resize function instantly, so we force a minimum of 1 pixel.
    if new_w < 1: 
        new_w = 1
        
    # 4. Perform the high-quality bicubic interpolation resize operation
    # cv2.resize expects the dimensional tuple argument formatted as (Width, Height)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    return resized

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ctc_greedy_decoder(preds, blank_index=BLANK_INDEX):
    decoded = []
    prev = blank_index
    for p in preds:
        val = p.item() if hasattr(p, 'item') else int(p)
        if val != prev and val != blank_index:
            decoded.append(val)
        prev = val
    return decoded

def calculate_ctc_loss(y_pred, labels, input_lengths, label_lengths):
    """
    Computes the Connectionist Temporal Classification (CTC) loss for sequence recognition.
    Safeguards against length crashes and automatically formats tensors for PyTorch.
    """
    # 1. Get the actual maximum sequence length outputted by the neural network
    # y_pred shape is [Batch, SequenceLength, NumClasses], so index 1 is our timeline steps
    max_output_seq_len = y_pred.size(1)
    
    # 2. CRITICAL CRASH PROTECTION: Check if any target text is longer than the network's output timeline
    # CTC cannot align text if the number of letters to predict is greater than the available output steps.
    # We check both individual sample input lengths and the total global sequence length limit.
    if (label_lengths > input_lengths).any() or (label_lengths > max_output_seq_len).any():
        raise ValueError(
            "CTC Failure: Target text length is longer than network sequence length. "
            "Ensure images aren't cut too narrow or single characters aren't heavily stretched."
        )
    
    # 3. Apply Log Softmax along the vocabulary class axis (dimension index 2)
    # PyTorch's native ctc_loss strictly requires log-probabilities as inputs rather than raw values
    y_pred_log = y_pred.log_softmax(2)
    
    # 4. Compute and return the final average mathematical loss value
    return torch.nn.functional.ctc_loss(
        # PyTorch ctc_loss expects the sequence dimension first: [SequenceLength, Batch, NumClasses]
        # transpose(0, 1) swaps the Batch (0) and SequenceLength (1) axes perfectly
        y_pred_log.transpose(0, 1), 
        labels,            # The flattened 1D tensor containing our target character indices
        input_lengths,     # Tensor tracking how long each model output timeline is
        label_lengths,     # Tensor tracking how many total characters are in each true text string
        blank=BLANK_INDEX, # Tell the loss function exactly which index represents our '<blank>' token
        reduction='mean',  # Average the final score across all images inside this specific training batch
        
        # LOW-VRAM SAFETY NET: If a highly blurred/faded image produces an impossible gradient anomaly,
        # zero_infinity forces PyTorch to set that specific bad sample loss to 0 instead of NaN.
        # This keeps your home GPU training loop moving forward safely without crashing mid-run!
        zero_infinity=True 
    )


def ctc_greedy_decoder_batch(preds, idx_to_char, blank=0):

    pred_texts = []
    for seq in preds:               # loop over batch
        decoded = []
        prev = blank
        for p in seq:               # loop over timesteps
            p = p.item() if hasattr(p, "item") else int(p)
            if p != prev and p != blank:
                decoded.append(idx_to_char.get(p, ''))
            prev = p
        pred_texts.append("".join(decoded))
    return pred_texts
