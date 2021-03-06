# NumberX: reinvent the numbers again through RL

Numeral system is one basic symbolic system of our humankind. Various study in anthropology and linguistics showed that
the natrual numeral system developed by human are sophisticated and diversed. Can AI reinvent the numbers again?

In the article [1], the author had given several conditions on a good numeral system.
And here, I give a modified version. A numeral system is required
* to be a complete covering under a limit
* not to have any ambiguity or homonymy
* to have few or no redundancy (considering the problem of 1 = 0.99999...)
* to have a construct method to extend the system further above any limit

The above conditions showed that a good formal definition of general numeral system is needed,
and Peano system is only one example.

Though we need to develop a formal definition of general numeral system, our perspective is not formal based on strings.
We propose a geometrically formal point of view for human knowledge.

We believe the invention history of numbers was driven by some natural optimization processes,
so now we can reinvent numbers again by RL.

The meaning of the symbols in a numeral system can be fully explained by some geometrical object which enjoy some minimal measurements.
That is a geometrical theory at meta level.

Current progress and todos
==========================
Progress

* We are still trying to formulate the problem into a proper form
* We proposed a gym env Serengeti that need an invention of numeral symbols and the ability of counting.

TODOs

* Develop a hierarchical RL env 

How to play
===========

In Bash:
```bash
git clone git@github.com:mountain/numberx.git
cd numberx
. hello
python -m main -g 0
```

In zsh
```zsh
git clone git@github.com:mountain/numberx.git
cd numberx
. ./hello
python -m main -g 0
```

How to contribute
=================

* formulate the problem into a proper form
* define a new simpler gym which still can capture the nature of numeral system
* give RL algorithms can solve the gym problems efficiently

FAQ
====

* What is the differences between [deepmind/mathematics_dataset](https://github.com/deepmind/mathematics_dataset) and this project?

[deepmind/mathematics_dataset](https://github.com/deepmind/mathematics_dataset) is a supervised learning task, 
while our gym environments try to define some unsupervised learning tasks in which the invention of numeral symbols and ability of counting are mandatory.

----

[1] James R Hurford: [ARTIFICIALLY GROWING A NUMERAL SYSTEM](http://www.lel.ed.ac.uk/~jim/grownum.html)







