"""Debug script to find which samples are assigned to each GPU."""

import math


def get_rank_indices(total_size, batch_size, num_replicas, rank):
    """Calculate the start and end sample indices for a given rank."""
    num_batches = math.ceil(total_size / batch_size)

    # Calculate which batches this rank handles
    batches_per_rank_base = num_batches // num_replicas
    remainder = num_batches % num_replicas

    # This rank gets an extra batch if rank < remainder
    batches_for_this_rank = batches_per_rank_base + (1 if rank < remainder else 0)

    # Calculate starting batch: base position + number of extra batches given to earlier ranks
    start_batch = rank * batches_per_rank_base + min(rank, remainder)
    end_batch = start_batch + batches_for_this_rank

    # Convert batch indices to sample indices
    start_idx = start_batch * batch_size
    end_idx = min(end_batch * batch_size, total_size)

    return start_idx, end_idx


def main():
    # Your dataset info - adjust these values
    total_images = 5737  # From your logs: "5737 images"
    batch_size = 64  # The batch size that causes freeze
    num_gpus = 4

    # Per-GPU batch size
    per_gpu_batch = batch_size // num_gpus  # 64 / 4 = 16

    print(f"Dataset: {total_images} images")
    print(f"Total batch size: {batch_size}")
    print(f"Per-GPU batch size: {per_gpu_batch}")
    print(f"Number of GPUs: {num_gpus}")
    print()

    for rank in range(num_gpus):
        start_idx, end_idx = get_rank_indices(total_images, per_gpu_batch, num_gpus, rank)
        print(f"GPU {rank} (rank {rank}): samples {start_idx} to {end_idx - 1} ({end_idx - start_idx} samples)")

    print()
    print("=" * 60)
    print("GPU 3 (last GPU) gets the LAST chunk of your dataset.")
    print("If the freeze only happens with 4 GPUs, check samples in GPU 3's range.")
    print()

    # Show the specific range for GPU 3
    start_idx, end_idx = get_rank_indices(total_images, per_gpu_batch, num_gpus, 3)
    print(f"Check images at indices {start_idx} to {end_idx - 1} for corruption or issues.")
    print()

    # Also show first few indices that GPU 3 will try to load
    print(f"First 10 sample indices for GPU 3: {list(range(start_idx, min(start_idx + 10, end_idx)))}")


if __name__ == "__main__":
    main()
