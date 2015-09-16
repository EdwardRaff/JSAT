package jsat.datatransform;

import jsat.classifiers.DataPoint;

/**
 * Creates a new Transform object that simply uses the inverse of an
 * {@link InvertibleTransform} as a regular transform. This allows one to apply
 * inverses after the fact in a simple matter like:
 * 
 * <pre>
 * <code>
 *  DataSet x = //some data set;
 *  InvertibleTransform transform = //some transform;
 *  x.applyTransform(transform);//apply the original transform
 *  //reverse the transform, getting back to where we started
 *  x.applyTransform(new InverseOfTransform(transform));
 * </code>
 * </pre>
 *
 * @author Edward Raff
 */
public class InverseOfTransform implements DataTransform {

  private static final long serialVersionUID = 2565737661260748018L;
  private final InvertibleTransform transform;

  /**
   * Copy constructor
   *
   * @param toClone
   *          the object to copy
   */
  public InverseOfTransform(final InverseOfTransform toClone) {
    this(toClone.transform.clone());
  }

  /**
   * Creates a new transform that uses the null null
   * {@link InvertibleTransform#transform(jsat.classifiers.DataPoint) transform}
   * of the given transform
   *
   * @param transform
   *          the transform to use the inverse function of
   */
  public InverseOfTransform(final InvertibleTransform transform) {
    this.transform = transform;
  }

  @Override
  public InverseOfTransform clone() {
    return new InverseOfTransform(this);
  }

  @Override
  public DataPoint transform(final DataPoint dp) {
    return transform.inverse(dp);
  }

}
