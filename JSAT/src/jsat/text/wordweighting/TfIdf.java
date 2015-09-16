package jsat.text.wordweighting;

import static java.lang.Math.log;
import java.util.List;
import jsat.linear.Vec;

/**
 * Applies Term Frequency Inverse Document Frequency (TF IDF) weighting to the
 * word vectors.
 *
 * @author Edward Raff
 */
public class TfIdf extends WordWeighting {

  public enum TermFrequencyWeight {
    /**
     * BOOLEAN only takes into account whether or not the word is present in the
     * document. <br>
     * 1.0 if the count is non zero.
     */
    BOOLEAN, /**
              * LOG returns a term weighting in [1, infinity) based on the log
              * value of the term frequency<br>
              * 1 + log(count)
              */
    LOG, /**
          * DOC_NORMALIZED returns a term weighting in [0, 1] by normalizing the
          * frequency by the most common word in the document. <br>
          * count/(most Frequent word in document)
          *
          */
    DOC_NORMALIZED;
  }

  private static final long serialVersionUID = 5749882005002311735L;

  private double totalDocuments;
  private List<Integer> df;
  private double docMax = 0.0;
  private final TermFrequencyWeight tfWeighting;

  /**
   * Creates a new TF-IDF document weighting scheme that uses
   * {@link TermFrequencyWeight#LOG LOG} weighting for term frequency.
   */
  public TfIdf() {
    this(TermFrequencyWeight.LOG);
  }

  /**
   * Creates a new TF-IDF document weighting scheme that uses the specified term
   * frequency weighting
   *
   * @param tfWeighting
   *          the weighting method to use for the term frequency (tf) component
   */
  public TfIdf(final TermFrequencyWeight tfWeighting) {
    this.tfWeighting = tfWeighting;
  }

  @Override
  public void applyTo(final Vec vec) {
    if (tfWeighting == TermFrequencyWeight.DOC_NORMALIZED) {
      docMax = vec.max();
    }
    vec.applyIndexFunction(this);
  }

  @Override
  public double indexFunc(final double value, final int index) {
    if (index < 0 || value == 0.0) {
      return 0.0;
    }

    double tf;// = 1+log(value);
    switch (tfWeighting) {
    case BOOLEAN:
      tf = 1.0;
      break;
    case LOG:
      tf = 1 + log(value);
      break;
    case DOC_NORMALIZED:
      tf = value / docMax;
      break;
    default:
      tf = value;
    }
    final double idf = log(totalDocuments / df.get(index));

    return tf * idf;
  }

  @Override
  public void setWeight(final List<? extends Vec> allDocuments, final List<Integer> df) {
    totalDocuments = allDocuments.size();
    this.df = df;
  }
}
