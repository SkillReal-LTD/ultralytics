#!/usr/bin/env python3
"""Debug script to check YOLO checkpoint epoch numbering"""
import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

print("\n=== Checkpoint Keys ===")
print(checkpoint.keys())
print(checkpoint.values())

print("\n=== Epoch Information ===")
if "epoch" in checkpoint:
    print(f"checkpoint['epoch'] = {checkpoint['epoch']}")
    print(f"Type: {type(checkpoint['epoch'])}")
else:
    print("No 'epoch' key found")

# Check for other epoch-related fields
for key in checkpoint.keys():
    if 'epoch' in key.lower():
        print(f"{key} = {checkpoint[key]}")

# Check best fitness info if available
if "best_fitness" in checkpoint:
    print(f"\nbest_fitness = {checkpoint['best_fitness']}")
