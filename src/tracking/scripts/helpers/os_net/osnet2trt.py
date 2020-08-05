import torchreid
import torch
import tensorrt as trt
import torch2trt
from torch2trt import TRTModule
import numpy as np

BATCH_SIZE = 8
OPTIMIZED_MODEL = 'osnet_trt.pth'
OPTIMIZED_MODEL_FP16 = 'osnet_trt_fp16.pth'

model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000,
                                     pretrained=True, use_gpu=True).cuda()

inputs = torch.empty(BATCH_SIZE, 3, 256, 128).uniform_().cuda()

model.eval()

print("\nCompiling TRT Model...")
model_trt = torch2trt.torch2trt(model, [inputs], log_level=trt.Logger.WARNING, max_batch_size=BATCH_SIZE,
                                fp16_mode=False, max_workspace_size=1<<27)
torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
print("Compiled and saved TRT Model!")
inputs = torch.empty(BATCH_SIZE, 3, 256, 128).uniform_().cuda()

print("\nCompiling FP16 TRT Model...")
model_trt_fp16 = torch2trt.torch2trt(model, [inputs], log_level=trt.Logger.WARNING, max_batch_size=BATCH_SIZE,
                                fp16_mode=True, max_workspace_size=1<<27)
torch.save(model_trt_fp16.state_dict(), OPTIMIZED_MODEL_FP16)
print("Compiled and saved TRT Model!")

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
model_trt_fp16 = TRTModule()
model_trt_fp16.load_state_dict(torch.load(OPTIMIZED_MODEL_FP16))

inputs = torch.empty(BATCH_SIZE, 3, 256, 128).uniform_().cuda()
inputs_trt = inputs.clone()
inputs_trt_fp16 = inputs.clone()
preds_torch = model(inputs).cpu().detach().numpy()
preds_trt = model_trt(inputs_trt).cpu().detach().numpy()
preds_trt_fp16 = model_trt_fp16(inputs_trt_fp16).cpu().detach().numpy()
        

print("DIFFERENCE: ", np.sum(np.abs(preds_torch - preds_trt)))
print("DIFFERENCE FP16: ", np.sum(np.abs(preds_torch - preds_trt_fp16)))