package jsat.text.tokenizer;

import java.util.List;

import jsat.text.stemming.Stemmer;

/**
 *
 * @author Edward Raff
 */
public class StemmingTokenizer implements Tokenizer {

  private static final long serialVersionUID = 2883247633791522390L;
  private final Stemmer stemmer;
  private final Tokenizer baseTokenizer;

  public StemmingTokenizer(final Stemmer stemmer, final Tokenizer baseTokenizer) {
    this.stemmer = stemmer;
    this.baseTokenizer = baseTokenizer;
  }

  @Override
  public List<String> tokenize(final String input) {
    final List<String> tokens = baseTokenizer.tokenize(input);
    stemmer.applyTo(tokens);
    return tokens;
  }

  @Override
  public void tokenize(final String input, final StringBuilder workSpace, final List<String> storageSpace) {
    baseTokenizer.tokenize(input, workSpace, storageSpace);
    stemmer.applyTo(storageSpace);
  }

}
