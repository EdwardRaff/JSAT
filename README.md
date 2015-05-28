# Java Statistical Analysis Tool

JSAT is a library for quickly getting started with Machine Learning problems. It is developed in my free time, and made available for use under the GPL 3. Part of the library is for self education, as such - all code is self contained. JSAT has no external dependencies, and is pure Java. I also aim to make the library suitably fast for small to medium size problems. As such, much of the code supports parallel execution.

## Get JSAT

You can download JSAT from my maven repo, add the below to your pom file

```
<repositories>
  <repository>
    <id>edwardraff-repo</id>
    <url>http://www.edwardraff.com/maven-repo/</url>
  </repository>
</repositories>

<dependencies>
  <dependency>
    <groupId>com.edwardraff</groupId>
    <artifactId>JSAT</artifactId>
    <version>0.0.1</version>
  </dependency>
</dependencies>
```

I will also host a snapshot directory, to access it - chang "maven-repo" to "maven-snapshot-repo" for the "<url>" tag. 

## Why use JSAT? 

For reasarch and specialized needs, JSAT has one of the largest collections of algorithms available in any framework. See an incomplete list [here](https://github.com/EdwardRaff/JSAT/wiki/Algorithms). 

Additional, there are unfortinatly not as many ML tools for Java as there are for other lanagues. Compared to Weka, JSAT is [usually faster](http://jsatml.blogspot.com/2015/03/jsat-vs-weka-on-mnist.html). 

If you want to use JSAT and the GPL is not something that will work for you, let me know and we can discus the issue.

NOTE: I'm currently in the progress of migrating from Google code and making some maintenance changes. 
