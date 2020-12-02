# Mechanisms of Action (MoA) Prediction

This is my take on another exciting bio <a href='https://www.kaggle.com/c/lish-moa/overview'>challenge on Kaggle</a>.

The <a href='https://clue.io/'>Connectivity Map</a>, a project within the Broad Institute of MIT and Harvard, the <a href='http://lish.harvard.edu/'>Laboratory for Innovation Science at Harvard (LISH)</a>, and the NIH Common Funds Library of Integrated Network-Based Cellular Signatures (LINCS), present this challenge with the goal of advancing drug development through improvements to MoA prediction algorithms.

<b>What is the Mechanism of Action (MoA) of a drug? And why is it important?</b>

In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.

<b>How do we determine the MoAs of a new drug?</b>

One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression or cell viability patterns of drugs with known MoAs.

<b>How to evaluate the accuracy of a solution?</b>

Based on the MoA annotations, the accuracy of solutions will be evaluated on the average value of the logarithmic loss function applied to each drug-MoA annotation pair.

<h2>Competition</h2>
The competition attracted quite a crowd - over 4300 teams made their submissions. The chosen metric packed all competitors tightly on the leaderboard: top 1 has the score 0.01599 while the last place in the bronze zone - 438th, had just 0.01614. 1000th, and 2000th places finished with 0.01627, and 0.01662 respectively.<br>
Considering the nature of the tabular data, we have binary multilable classiffication problem. Many of competitors focused solely on new to me TabNet approach, while others tried to exploit trusted neural nets. <br>
My first attempt was a simple shallow neural net with 3 fully connected layer (NN1.py). It is super fast in comparison to more complex and advanced approaches, however gives somewhat decent score of 0.01688. 
