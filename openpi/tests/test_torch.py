import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import os

def setup_distributed(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    return dist.is_initialized()

def test_distributed_tensor(rank, world_size):
    """Test distributed tensor operations"""
    try:
        # Create tensor on each process
        tensor = torch.ones(2, 2) * rank
        print(f"Process {rank}: Initial tensor =\n{tensor}")

        # All-reduce operation
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(i * torch.ones(2, 2) for i in range(world_size))
        is_correct = torch.allclose(tensor, expected_sum)
        print(f"Process {rank}: After all_reduce =\n{tensor}")
        print(f"Process {rank}: Result is correct: {is_correct}")
        
        return is_correct
    except Exception as e:
        print(f"Process {rank}: Error in distributed tensor test: {e}")
        return False

def check_pytorch_installation(rank=None, world_size=None):
    """Check PyTorch installation including distributed capabilities"""
    print("PyTorch Installation Check")
    print("==========================")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print("\nCUDA Information:")
        print("-----------------")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device ID: {current_device}")
        device = torch.device(f'cuda:{current_device}')
        print(f"Current CUDA device: {device}")
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA device name: {device_name}")
        
        print("\nTensor Operations:")
        print("------------------")
        try:
            x = torch.tensor(1).to(device)
            print(f"Moved tensor: {x}")
            y = torch.tensor(1, device=device)
            print(f"Created tensor: {y}")
            if x.device.type == 'cuda' and y.device.type == 'cuda':
                print("\nSuccess: PyTorch is correctly installed with CUDA support!")
            else:
                print("\nWarning: Tensors were not properly moved to CUDA devices.")
        except Exception as e:
            print(f"\nError during tensor operations: {e}")
    else:
        print("\nNote: CUDA is not available. PyTorch will run on CPU only.")

    # Distributed Training Test
    if rank is not None:
        print(f"\nDistributed Training Test (Process {rank}):")
        print("----------------------------------------")
        try:
            is_initialized = setup_distributed(rank, world_size)
            print(f"Process {rank}: Distributed setup successful: {is_initialized}")
            
            if is_initialized:
                test_result = test_distributed_tensor(rank, world_size)
                if test_result:
                    print(f"Process {rank}: Distributed operations test passed!")
                else:
                    print(f"Process {rank}: Distributed operations test failed!")
                
                # Cleanup
                dist.destroy_process_group()
        except Exception as e:
            print(f"Process {rank}: Error in distributed setup: {e}")

    print("\nSystem Information:")
    print("-------------------")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {sys.platform}")

def run_distributed_test(world_size=2):
    """Launch distributed test with multiple processes"""
    mp.spawn(check_pytorch_installation,
            args=(world_size,),
            nprocs=world_size,
            join=True)

if __name__ == "__main__":
    # Run regular installation check
    print("Running single-process installation check...")
    check_pytorch_installation()
    
    print("\n\nRunning distributed training check...")
    run_distributed_test(world_size=2)
    
    print('\n\nAll tests passed.')