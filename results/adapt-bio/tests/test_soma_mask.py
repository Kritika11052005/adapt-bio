import torch
from adapt_bio.core.soma_mask import SOMAMask

def test_soma_mask_updates():
    num_heads, seq_len, k = 2, 8, 3
    anticipation_steps = 5
    rna_update_interval = 10
    
    soma = SOMAMask(
        num_heads=num_heads,
        seq_len=seq_len,
        k=k,
        anticipation_steps=anticipation_steps,
        rna_update_interval=rna_update_interval
    )
    
    # Step 0 to 9: should stay dense (since update interval is 10)
    for step in range(9):
        fake_weights = torch.rand(num_heads, seq_len, seq_len)
        mask = soma(fake_weights, step=step)
        assert torch.all(mask)  # all True (dense)
        
    # Step 10: should update and trigger Starling k sparsity
    fake_weights = torch.rand(num_heads, seq_len, seq_len)
    mask = soma(fake_weights, step=10)
    
    # Check that it's no longer dense and respects k=3
    assert not torch.all(mask)
    assert torch.all(mask.sum(dim=-1) == k)
