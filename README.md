# HCXA
Histogram-based Count Maximization Algorithm.

HXCA is a online exploration tool that allows for choosing new reference state while taking previously visited states into account. HCXA discretizes the continuous state space and uses histograms as N-dimensional counter tables.As the continuous state space is approximated by its discrete counter part, the accuracy of this representation depends on resolution which in turn depends on the number of bins used in histogram to depict each dimension. Higher number of bins results in better resolution and finer approximation of the continuous space.By utilizing the Merge Sort algorithm, HCXA effectively explores the state-space, prioritizing regions that are underrepresented in the current sample set. This approach ensures a more balanced and accurate representation of the reference coverage function. The sorted list is then traversed in a first-in-first-out (FIFO) fashion and a validity check is performed.
```python
print('Hello')
```
