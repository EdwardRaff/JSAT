package jsat.datatransform;

import jsat.DataSet;
import jsat.classifiers.DataPoint;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * A transformation to shift all numeric variables so that their mean is zero
 *
 * @author Edward Raff
 */
public class ZeroMeanTransform implements InPlaceInvertibleTransform {

  private static final long serialVersionUID = -7411115746918116163L;
  /**
   * Shift vector stores the mean value of each variable in the original data set.
   */
  private Vec shiftVector;

  public ZeroMeanTransform(DataSet dataset) {
    shiftVector = new DenseVector(dataset.getNumNumericalVars());
    shiftVector = dataset.getColumnMeanVariance()[0];
  }

  /**
   * Copy constructor
   *
   * @param other the transform to make a copy of
   */
  private ZeroMeanTransform(ZeroMeanTransform other) {
    this.shiftVector = other.shiftVector.clone();
  }

  @Override
  public DataPoint transform(DataPoint dp) {
    DataPoint newDP = dp.clone();
    mutableTransform(newDP);
    return newDP;
  }

  @Override
  public void mutableInverse(DataPoint dp) {
    dp.getNumericalValues().mutableAdd(shiftVector);
  }

  @Override
  public DataPoint inverse(DataPoint dp) {
    DataPoint newDP = dp.clone();
    mutableInverse(dp);
    return newDP;
  }

  @Override
  public void mutableTransform(DataPoint dp) {
    dp.getNumericalValues().mutableSubtract(shiftVector);
  }

  @Override
  public boolean mutatesNominal() {
    return false;
  }

  @Override
  public ZeroMeanTransform clone() {
    return new ZeroMeanTransform(this);
  }

  /**
   * Factory for producing new {@link ZeroMeanTransform} transforms.
   */
  static public class ZeroMeanTransformFactory implements DataTransformFactory {

    public ZeroMeanTransformFactory() {
    }

    @Override
    public DataTransform getTransform(DataSet dataset) {
      return new ZeroMeanTransform(dataset);
    }

    @Override
    public ZeroMeanTransformFactory clone() {
      return new ZeroMeanTransformFactory();
    }

  }
}
