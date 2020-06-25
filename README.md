# FPPDL
code for TPDS paper "Towards Fair and Privacy-Preserving Federated Deep Models"! Folder "dpgan" is used to generate DPGAN samples on each party!

# How to run:
th fppdl_tpds.lua -dataset mnist -model deep -slevel 1 -imbalanced 1 -netSize 4 -nepochs 100 -local_nepochs 5 -batchSize 10 -learningRate 0.15 -taskID mnist_deep_p4e100_imbalanced -shardID mnist_p4_imbalanced -run run1 -pretrain 1 -credit_fade 1

# How to analyze fairness:
All logs will be dumped into folder "logs". Process log and analyze fairness as follows:
```
1. X axis: standalone accuracy 
grep "standalone" logs/fppdl_mnist_deep_p4e100_slevel01_imbalanced_IID1_pretrain1_localepoch5_localbatch10_lr0.15_run1_tpds.log >1.log
awk '{print $NF}' ORS=', ' 1.log
2. Y axis: final accuracy 
grep "final test acc" logs/fppdl_mnist_deep_p4e100_slevel01_imbalanced_IID1_pretrain1_localepoch5_localbatch10_lr0.15_run1_tpds.log >1.log
awk '{print $NF}' ORS=', ' 1.log
3. Finally, using scipy.stats.pearsonr(x,y) to calculate fairness.
```

# Requirements:
- torch7, download from http://torch.ch/
- python3

# Bibtex
Remember to cite the following papers if you use any part of the code:
```
@article{lyu2020towards,
  title={Towards Fair and Privacy-Preserving Federated Deep Models},
  author={Lyu, Lingjuan and Yu, Jiangshan and Nandakumar, Karthik and Li, Yitong and Ma, Xingjun and Jin, Jiong and Yu, Han and Ng, Kee Siong},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  volume={31},
  number={11},
  pages={2524--2541},
  year={2020},
  publisher={IEEE}
}
```
