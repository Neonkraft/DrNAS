import sys
sys.path.append("/work/dlclarge2/krishnan-tanglenas/DrNAS/")

import time
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from model_search import Network

def create_model(C=16, num_classes=10, layers=8):
    return Network(
        C,
        num_classes,
        layers,
        torch.nn.CrossEntropyLoss(),
        k=1,
    )

def profile_cell(search_model, cell, cell_input, n_repeat=10):
    search_model(torch.randn(1, 3, 32, 32).cuda())

    s0 = s1 = cell_input.cuda()
    weights = torch.randn(14, 9).cuda()

    start_time = time.time()
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for i in range(n_repeat):
                s0, s1 = s1, cell(s0[:, :256, ...], s1[:, :256, ...], weights)

    end_time = time.time()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
    print(f"Time taken: {(end_time - start_time) / n_repeat}")
    prof.export_chrome_trace(f"original_drnas_lastcell_forward_bs{cell_input.shape[0]}_{n_repeat}times.json")

def profile_model(search_model, n_repeat=10):
    search_model(torch.randn(1, 3, 32, 32).cuda())

    input_data = torch.randn(64, 3, 32, 32).cuda()
    start_time = time.time()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(n_repeat):
                out = search_model(input_data)

    end_time = time.time()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(f"Time taken: {(end_time - start_time) / n_repeat}")

    prof.export_chrome_trace(f"profiles/original_drnas_model_8cells_16channels_forward_bs64_{n_repeat}times.json")

    return prof

if __name__ == "__main__":

    # Our Code WS
    search_model = create_model().cuda()
    search_model.train()
    # cell = search_model.cells[-1]

    # profile_cell(search_model, cell, torch.randn(64, 256, 8, 8), n_repeat=10)

    profile_model(search_model, n_repeat=10)

