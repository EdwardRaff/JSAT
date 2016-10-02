# Java Statistical Analysis Tool

[![Join the chat at https://gitter.im/EdwardRaff/JSAT](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/EdwardRaff/JSAT?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

JSAT is a library for quickly getting started with Machine Learning problems. It is developed in my free time, and made available for use under the GPL 3. Part of the library is for self education, as such - all code is self contained. JSAT has no external dependencies, and is pure Java. I also aim to make the library suitably fast for small to medium size problems. As such, much of the code supports parallel execution.

## Get JSAT

You can download JSAT from maven central, add the below to your pom file

```xml
<dependencies>
  <dependency>
    <groupId>com.edwardraff</groupId>
    <artifactId>JSAT</artifactId>
    <version>0.0.6</version>
  </dependency>
</dependencies>
```

I will also host a snapshot directory, to access it - change "maven-repo" to "maven-snapshot-repo" for the "\<url>" tag. 

## Why use JSAT? 

For reasarch and specialized needs, JSAT has one of the largest collections of algorithms available in any framework. See an incomplete list [here](https://github.com/EdwardRaff/JSAT/wiki/Algorithms). 

Additional, there are unfortunately not as many ML tools for Java as there are for other lanagues. Compared to Weka, JSAT is [usually faster](http://jsatml.blogspot.com/2015/03/jsat-vs-weka-on-mnist.html). 

If you want to use JSAT and the GPL is not something that will work for you, let me know and we can discuss the issue.

See the [wiki](https://github.com/EdwardRaff/JSAT/wiki) for more information as well as some examples on how to use JSAT. 

## Note

Updates to JSAT may be slowed as I begin a PhD program in Computer Science. The project isn’t abandoned! I just have limited free time, and will be balancing my PhD work with a full time job. If you discover more hours in the day, please let me know! Development will be further slowed due to some health issues. I had to spend much of my time recently on this but am begining to improve. I'll continue to try and be prompt on any bug reports and emails, but new features will be a bit slower. 


