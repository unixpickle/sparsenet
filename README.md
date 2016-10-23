# sparsenet

This is an attempt to implement an effective sparse neural network. The goal is to mimic the connectivity of the brain, where spatially nearby neurons are more likely to connect with each other.

My hope is that sparse networks with properly localized connections will learn to specialize--that is, different spatial areas in the network will learn different sub-tasks. Sparse networks could provide a natural way to parallelize large neural networks. They could also make it possible to have networks with a vast number of neurons without requiring too much graphics memory during training.
