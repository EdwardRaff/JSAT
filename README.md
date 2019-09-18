# Java Statistical Analysis Tool

<a href='https://travis-ci.org/EdwardRaff/JSAT/builds'><img src='https://travis-ci.org/EdwardRaff/JSAT.svg?branch=master'></a>


JSAT is a library for quickly getting started with Machine Learning problems. It is developed in my free time, and made available for use under the GPL 3. Part of the library is for self education, as such - all code is self contained. JSAT has no external dependencies, and is pure Java. I also aim to make the library suitably fast for small to medium size problems. As such, much of the code supports parallel execution.

The current master branch of JSAT is going through a larger refactoring as JSAT moves to Java 8. This may cause some examples to break if used against the head version, but they should be fixible with minimal changes.

## Get JSAT

Ther current release of JSAT is version 0.0.9, and supports Java 6. The current master branch is now Java 8+. 

You can download JSAT from maven central, add the below to your pom file

```xml
<dependencies>
  <dependency>
    <groupId>com.edwardraff</groupId>
    <artifactId>JSAT</artifactId>
    <version>0.0.9</version>
  </dependency>
</dependencies>
```

If you want to use the bleeding edge, but don't want to bother building yourself, I recommend you look at [jitpack.io](https://jitpack.io/#EdwardRaff/JSAT). It can build a POM repo for you for any specific commit version. Click on "Commits" in the link and then click "get it" for the commit version you want. 

If you want to read the javadoc's online, you can find them hosted [on my website here](http://www.edwardraff.com/jsat_docs/JSAT-0.0.8-javadoc/). 

## Why use JSAT? 

For research and specialized needs, JSAT has one of the largest collections of algorithms available in any framework. See an incomplete list [here](https://github.com/EdwardRaff/JSAT/wiki/Algorithms). 

Additional, there are unfortunately not as many ML tools for Java as there are for other languages. Compared to Weka, JSAT is [usually faster](http://jsatml.blogspot.com/2015/03/jsat-vs-weka-on-mnist.html). 

If you want to use JSAT and the GPL is not something that will work for you, let me know and we can discuss the issue.

See the [wiki](https://github.com/EdwardRaff/JSAT/wiki) for more information as well as some examples on how to use JSAT. 

## Note

Updates to JSAT may be slowed as I begin a PhD program in Computer Science. The project isn’t abandoned! I just have limited free time, and will be balancing my PhD work with a full time job. If you discover more hours in the day, please let me know! Development will be further slowed due to some health issues. I'll continue to try and be prompt on any bug reports and emails, but new features will be a bit slower. Please use the github issues first for contact. 

## Citations

If you use JSAT and find it helpful, citations are appreciated! Please cite the [JSAT paper](http://www.jmlr.org/papers/v18/16-131.html) published at JMLR. If you're feeling a little lazy, the bibtex is below:

```
@article{JMLR:v18:16-131,
author = {Raff, Edward},
journal = {Journal of Machine Learning Research},
number = {23},
pages = {1--5},
title = {JSAT: Java Statistical Analysis Tool, a Library for Machine Learning},
url = {http://jmlr.org/papers/v18/16-131.html},
volume = {18},
year = {2017}
}
```
