# Best Result: exp_20c

- **Performance**: 0.764ms at H=448
- **Improvement**: -83.7% vs baseline (4.695ms), 6.14x speedup
- **Strategy**: vload3 + block_base + WG=8 + manual unroll ph=2 + explicit t=2
- **Theoretical minimum**: ~0.69ms (memory bandwidth bound)
- **Gap to theoretical**: 10.7%
- **All sizes PASS**: H=128 (0.095ms), H=224 (0.230ms), H=320 (0.414ms), H=448 (0.764ms), H=544 (1.112ms), H=672 (1.674ms), H=800 (2.347ms), H=896 (2.915ms)
