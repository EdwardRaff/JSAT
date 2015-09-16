package jsat.text;

import java.util.ArrayList;
import java.util.List;
import jsat.linear.SparseVector;
import jsat.linear.Vec;
import jsat.text.tokenizer.Tokenizer;
import jsat.text.wordweighting.WordWeighting;

/**
 * Text Vector creator to that uses hashed features.
 *
 * @author Edward Raff
 */
public class HashedTextVectorCreator implements TextVectorCreator {

  private static final long serialVersionUID = 1081388790985568192L;
  private final int dimensionSize;
  private final Tokenizer tokenizer;
  private final WordWeighting weighting;

  /**
   * Creates a new text vector creator that works with hash-trick features
   *
   * @param dimensionSize
   *          the dimension size of the feature space
   * @param tokenizer
   *          the tokenizer to apply to incoming strings
   * @param weighting
   *          the weighting process to apply to each loaded document.
   */
  public HashedTextVectorCreator(final int dimensionSize, final Tokenizer tokenizer, final WordWeighting weighting) {
    if (dimensionSize <= 1) {
      throw new ArithmeticException("Vector dimension must be a positive value");
    }
    this.dimensionSize = dimensionSize;
    this.tokenizer = tokenizer;
    this.weighting = weighting;
  }

  @Override
  public Vec newText(final String input) {
    return newText(input, new StringBuilder(), new ArrayList<String>());
  }

  @Override
  public Vec newText(final String input, final StringBuilder workSpace, final List<String> storageSpace) {
    tokenizer.tokenize(input, workSpace, storageSpace);
    final SparseVector vec = new SparseVector(dimensionSize);
    for (final String word : storageSpace) {
      // XXX This code generates a hashcode and then computes the absolute value
      // of that hashcode. If the hashcode is Integer.MIN_VALUE, then the result
      // will be negative as well (since Math.abs(Integer.MIN_VALUE) ==
      // Integer.MIN_VALUE).
      vec.increment(Math.abs(word.hashCode()) % dimensionSize, 1.0);
    }
    weighting.applyTo(vec);
    return vec;
  }
}
